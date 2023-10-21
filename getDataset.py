import  torch
import  os, glob
import  random, csv
from    torch.utils.data import Dataset, DataLoader
from    torchvision import transforms,datasets
from    PIL import Image
import numpy


class MultipleApply:
    """Apply a list of transformations to an image and get multiple transformed images.

    Args:
        transforms (list or tuple): list of transformations

    Example:

        >>> transform1 = T.Compose([
        ...     ResizeImage(256),
        ...     T.RandomCrop(224)
        ... ])
        >>> transform2 = T.Compose([
        ...     ResizeImage(256),
        ...     T.RandomCrop(224),
        ... ])
        >>> multiply_transform = MultipleApply([transform1, transform2])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return [t(image) for t in self.transforms]

class getDataset(Dataset):

    def __init__(self, root, resize, mode,transform=None,s_t=None,is_labeled=None):
        super(getDataset, self).__init__()

        self.root = root
        self.resize = resize
        self.transform=transform
        # print(self.transform)

        self.name2label = {} # "sq...":0
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        # image, label
        self.images, self.labels = self.load_csv('images.csv')
        #
        #
        source_image_path = []
        source_image_label = []

        target_image_label = []
        target_image_path = []
        if mode=='train': # 70%
            self.images = self.images[:int(0.7 * len(self.images))]
            self.labels = self.labels[:int(0.7 * len(self.labels))]
            source_normal_counts=0
            source_pneumonia_counts=0
            if s_t=='source':
                for i in range(len(self.images)):
                    if self.labels[i]<1 and source_normal_counts<5613:
                        source_image_path.append(self.images[i])
                        source_image_label.append(self.labels[i])
                        source_normal_counts+=1
                    if self.labels[i]==2 and source_pneumonia_counts<2306:
                        self.labels[i]=1
                        source_image_path.append(self.images[i])
                        source_image_label.append(self.labels[i])
                        source_pneumonia_counts+=1
                self.images=source_image_path
                self.labels=source_image_label
            if s_t=='target':
                target_normal_counts = 0
                target_covid_counts = 0
                for i in range(len(self.images)):
                    if self.labels[i] ==1 and target_covid_counts<258:
                        target_image_path.append(self.images[i])
                        target_image_label.append(self.labels[i])
                        target_covid_counts+=1
                    if self.labels[i] <1 and target_normal_counts<2541:
                        target_image_path.append(self.images[i])
                        target_image_label.append(self.labels[i])
                        target_normal_counts+=1
                self.images = target_image_path
                self.labels = target_image_label

        if mode=='val':
            self.images = self.images[int(0.7 * len(self.images)):int(0.85 * len(self.images))]
            self.labels = self.labels[int(0.7 * len(self.labels)):int(0.85 * len(self.labels))]
            for i in range(len(self.images)):
                if self.labels[i]<2:
                    target_image_path.append(self.images[i])
                    target_image_label.append(self.labels[i])
            self.images=target_image_path
            self.labels=target_image_label
        elif mode=='test':
            self.images = self.images[int(0.85*len(self.images)):]
            self.labels = self.labels[int(0.85*len(self.labels)):]
            normal_counts=0
            covid_counts=0
            for i in range(len(self.images)):
                if self.labels[i]<2:
                    if self.labels[i]==0 and normal_counts<885:
                        target_image_path.append(self.images[i])
                        target_image_label.append(self.labels[i])
                        normal_counts+=1
                    if self.labels[i]==1 and covid_counts<60:
                        target_image_path.append(self.images[i])
                        target_image_label.append(self.labels[i])
                        covid_counts+=1
            self.images=target_image_path
            self.labels=target_image_label

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))    #加载路径下的所有.jpeg图片

            print(len(images), images)
            print(images)
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]

                    writer.writerow([img, label])
                print('writen into csv file:', filename)
        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:

                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img=Image.open(img).convert('RGB')

        if self.transform is not None:
            img=self.transform(img)
        label = torch.tensor(label)
        return img, label
