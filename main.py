import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
from typing import Tuple

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from rocCurve import draw_roc_cur
from def_loss import ImageClassifierHead, entropy, classifier_discrepancy,analyse,multi_analyse,consistency_loss,js_div
import models as models
from sklearn.preprocessing import StandardScaler
from transforms import ResizeImage
from utils.data import ForeverDataIterator
from utils.metric import accuracy, ConfusionMatrix
from utils.meter import AverageMeter, ProgressMeter
from utils.logger import CompleteLogger
from utils.analysis import collect_feature, tsne, a_distance
from getDataset import getDataset,MultipleApply
import numpy as np
from randaugment import RandAugment
from discri import DomainDiscriminator,DomainAdversarialLoss,Bottleneck

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    cudnn.benchmark = True

    # Data loading code
    normalize = T.Normalize((0.4859,0.4859,0.4859),(0.0820,0.0820,0.0820))
    train_transform = T.Compose([
        ResizeImage(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
        ])
    transforms_train_strong = T.Compose([
        ResizeImage((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        RandAugment(2, 10),
        T.ToTensor(),
        normalize
    ])
    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    train_source_dataset = getDataset(args.path_source, args.image_size, 'train',
                                      transform=train_transform, s_t='source')

    train_target_dataset = getDataset(args.path_target, args.image_size, 'train',
                                      transform=MultipleApply([train_transform,transforms_train_strong]), s_t='target')
    val_dataset = getDataset(args.path_target, args.image_size, 'val',
                             transform=val_transform)
    test_dataset = getDataset(args.path_target, args.image_size, 'test',
                              transform=val_transform)

    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    G = models.__dict__[args.arch](pretrained=True).to(device)  # feature extractor
    num_classes = 2
    # two image classifier heads
    F1 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim).to(device)
    F2 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim).to(device)
    bottleneck=Bottleneck(G.out_features).to(device)
    domain_discri = DomainDiscriminator(in_feature=256, hidden_size=1024).to(device)

    # define optimizer
    # the learning rate is fixed according to origin paper
    optimizer_g = SGD([ 
        {"params": G.parameters()},
    ], lr=args.lr, weight_decay=0.0005)
    optimizer_f = SGD([
        {"params": F1.parameters()},
        {"params": F2.parameters()},
    ], momentum=0.9, lr=args.lr, weight_decay=0.0005)
    optimizer_d=SGD([
        {"params":G.parameters()},
        {'params':domain_discri.parameters()},
    ],momentum=0.9,lr=args.lr,weight_decay=0.0005)


    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        G.load_state_dict(checkpoint['G'])
        F1.load_state_dict(checkpoint['F1'])
        F2.load_state_dict(checkpoint['F2'])

    if args.phase == 'test':
        acc1,acc2,F1,precision,recall = validate(test_loader, G, F1, F2, args)
        print((acc1+acc2)/2,F1,precision,recall)
        return

    # start training
    best_f1=0
    Cri_CE_noreduce = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, G, F1, F2,bottleneck,domain_adv, optimizer_g, optimizer_f,optimizer_d, epoch, args,Cri_CE_noreduce)
    
        # evaluate on validation set
        results1,results2,f1,precision,recall = validate(val_loader, G, F1, F2, args)
        torch.save({
            'G': G.state_dict(),
            'F1': F1.state_dict(),
            'F2': F2.state_dict()
        }, logger.get_checkpoint_path('latest'))
        if f1 > best_f1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_f1=f1

    #evaluate on test set
    checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
    G.load_state_dict(checkpoint['G'])
    F1.load_state_dict(checkpoint['F1'])
    F2.load_state_dict(checkpoint['F2'])
    Results1, Results2,F1, Precision, Recall = validate(test_loader, G, F1, F2, args)
    print("test_acc: %.04f,test_f1: %.04f,test_precision: %.04f,test_recall: %.04f" % ((Results1+Results2)/2, F1,Precision,Recall))
    logger.close()

def train(train_source_iter, train_target_iter,G, F1, F2,bottleneck,domain_adv,
          optimizer_g, optimizer_f,optimizer_d, epoch, args,Cri_CE_noreduce):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

      # switch to train mode
    G.train()
    bottleneck.train()
    F1.train()
    F2.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        (x_t,x_t_strong),labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        x_t_strong=x_t_strong.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        x = torch.cat((x_s, x_t), dim=0)
        x_strong=torch.cat((x_s, x_t_strong), dim=0)
        assert x.requires_grad is False

        # measure data loading time
        data_time.update(time.time() - end)

        # Step A train all networks to minimize loss on source domain
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        g = G(x)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        loss=F.cross_entropy(y1_s, labels_s)+F.cross_entropy(y2_s,labels_s)+0.01*(entropy(y1_t) + entropy(y2_t))
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        # Step B train classifier to maximize discrepancy
        optimizer_f.zero_grad()
        g = G(x)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        loss = F.cross_entropy(y1_s, labels_s)+F.cross_entropy(y2_s,labels_s)+0.01*(entropy(y1_t) + entropy(y2_t))- classifier_discrepancy(y1_t, y2_t) * args.beta_weight
        loss.backward()
        optimizer_f.step()
        
        # # # Step C train genrator to minimize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        g = G(x)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)
        g_strong = G(x_strong)
        y_1_strong = F1(g_strong)
        y_2_strong = F2(g_strong)
        y1_s_strong, y1_t_strong = y_1_strong.chunk(2, dim=0)
        y2_s_strong, y2_t_strong = y_2_strong.chunk(2, dim=0)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        ssl_loss1, mask1 = consistency_loss(y1_t,
                                            y1_t_strong,
                                            Cri_CE_noreduce, 1.0, 0.95,
                                            use_hard_labels=True)  # consistency loss

        ssl_loss2, mask2 = consistency_loss(y2_t,
                                            y2_t_strong,
                                            Cri_CE_noreduce, 1.0, 0.95,
                                            use_hard_labels=True)  # consistency loss
        loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + \
               0.01*(entropy(y1_t) + entropy(y2_t))+classifier_discrepancy(y1_t, y2_t)* args.beta_weight+args.ssl_weight*(ssl_loss1+ssl_loss2)
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        #transfer adv loss
        optimizer_d.zero_grad()
        g = G(x)
        g_=bottleneck(g)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        g_s,g_t=g_.chunk(2,dim=0)
        transfer_loss = domain_adv(g_s,g_t)
        adv_loss= args.alpha_weight*transfer_loss
        adv_loss.backward()
        optimizer_d.step()


        cls_acc = accuracy(y1_s, labels_s)[0]
        tgt_acc = accuracy(y1_t, labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))
        trans_losses.update(transfer_loss.item(), x_t.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, G, F1,
             F2,args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1_1 = AverageMeter('Acc_1', ':6.2f')
    top1_2 = AverageMeter('Acc_2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1_1, top1_2],
        prefix='Test: ')

    # switch to evaluate mode
    G.eval()
    F1.eval()
    F2.eval()

    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None
    pro_list=[]
    gt_list=[]
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            gt_list.append(target.cpu().numpy())

            # compute output
            g = G(images)
            y1, y2 = F1(g), F2(g)
            y=(y1+y2)/2
            acc1, = accuracy(y1, target)
            acc2, = accuracy(y2, target)
            if confmat:
                confmat.update(target, y1.argmax(1))
            pro_list.append(torch.max(y,1)[1].cpu().numpy())
            top1_1.update(acc1.item(), images.size(0))
            top1_2.update(acc2.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
        pro_list = np.concatenate(pro_list)
        gt_list = np.concatenate(gt_list)
        F1, Pre, Recall = multi_analyse(gt_list, pro_list)
        print(' *ACC:{:4f}'.format((acc1+acc2)/2))
        print(' *F1:{:4f}'.format(F1))
        print(' *Precision:{:4f}'.format(Pre))
        print(' *Recall:{:4f}'.format(Recall))
        if confmat:
            print(confmat.format(classes))

    return top1_1.avg, top1_2.avg,F1,Pre,Recall


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )


    parser = argparse.ArgumentParser(description='Our model for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('--path_source', default='./data/COVID-19_Radiography_Dataset', help='source domain(s)')
    parser.add_argument('--path_target', default='./data/COVID-19_Radiography_Dataset', help='target domain(s)')
    parser.add_argument('--center-crop', default=False, action='store_true',
                        help='whether use center crop during training')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int)

    parser.add_argument('--alpha_weight', default=0.01, type=float,
                        help='the trade-off hyper-parameter for domain loss')
    parser.add_argument('--beta_weight', default=0.01, type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--ssl_weight', default=0.1, type=float,
                        help='the trade-off hyper-parameter for domain loss')

    parser.add_argument('--num-k', type=int, default=2, metavar='K',
                        help='how many steps to repeat the generator update')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 24)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='savingModels',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--image_size', type=int, default=224, help='image resolution')
    args = parser.parse_args()
    main(args)

