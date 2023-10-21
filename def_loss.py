from typing import Optional
import torch.nn as nn
import torch
import sklearn.metrics as metrics
from  sklearn.metrics import f1_score,precision_score,recall_score
import torch.nn.functional as F

def classifier_discrepancy(predictions1: torch.Tensor, predictions2: torch.Tensor) -> torch.Tensor:
    r"""The `Classifier Discrepancy` in
    `Maximum Classiﬁer Discrepancy for Unsupervised Domain Adaptation (CVPR 2018) <https://arxiv.org/abs/1712.02560>`_.

    The classfier discrepancy between predictions :math:`p_1` and :math:`p_2` can be described as:

    .. math::
        d(p_1, p_2) = \dfrac{1}{K} \sum_{k=1}^K | p_{1k} - p_{2k} |,

    where K is number of classes.

    Args:
        predictions1 (torch.Tensor): Classifier predictions :math:`p_1`. Expected to contain raw, normalized scores for each class
        predictions2 (torch.Tensor): Classifier predictions :math:`p_2`
    """
    return torch.mean(torch.abs(predictions1 - predictions2))


def entropy(predictions: torch.Tensor) -> torch.Tensor:
    r"""Entropy of N predictions :math:`(p_1, p_2, ..., p_N)`.
    The definition is:

    .. math::
        d(p_1, p_2, ..., p_N) = -\dfrac{1}{K} \sum_{k=1}^K \log \left( \dfrac{1}{N} \sum_{i=1}^N p_{ik} \right)

    where K is number of classes.

    .. note::
        This entropy function is specifically used in MCD and different from the usual :meth:`~dalib.modules.entropy.entropy` function.

    Args:
        predictions (torch.Tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
    """
    return -torch.mean(torch.log(torch.mean(predictions, 0) + 1e-6))


class ImageClassifierHead(nn.Module):
    r"""Classifier Head for MCD.

    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024

    Shape:
        - Inputs: :math:`(minibatch, F)` where F = `in_features`.
        - Output: :math:`(minibatch, C)` where C = `num_classes`.
    """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 1024):
        super(ImageClassifierHead, self).__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs)
def analyse(gt_list, p_list):
    p_list[p_list >= 0.5] = 1
    p_list[p_list < 0.5] = 0
    t_open, f_narrow, f_open, t_narrow = metrics.confusion_matrix(gt_list, p_list).ravel()
    F1 = f1_score(gt_list, p_list)
    accuracy = (t_narrow + t_open) / (t_narrow + t_open + f_narrow + f_open)
    precision = t_narrow / (t_narrow + f_narrow)
    recall = t_narrow / (t_narrow + f_open)
    return F1,precision,recall
def multi_analyse(gt_list, p_list):
    gt_list=gt_list.T
    p_list=p_list.T
    macro_f1=f1_score(gt_list,p_list,average="macro")
    precision=precision_score(gt_list,p_list,average="macro")
    recall=recall_score(gt_list,p_list,average="macro")
    return macro_f1,precision,recall
def consistency_loss(logits_w, logits_s,Cri_CE_noreduce,T=1.0, p_cutoff=0.0,
                     use_hard_labels=True):
    logits_w = logits_w.detach()
    pseudo_label = torch.softmax(logits_w / T, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    mask_binary = max_probs.ge(p_cutoff)
    mask = mask_binary.float()
    # if mask.mean().item() == 0:
    #     acc_selected = 0
    # else:
    #     acc_selected = (target_gt_for_visual[mask_binary].float() == max_idx[mask_binary]).float().mean().item()

    if use_hard_labels:
        masked_loss =Cri_CE_noreduce(logits_s, max_idx) * mask
    else:
        raise NotImplementedError
        # pseudo_label = torch.softmax(logits_w / T, dim=-1)
        # masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
    return masked_loss.mean(), mask.mean()
def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2