from __future__ import print_function

from PIL import Image
import torchvision
import torchvision.datasets as datasets
import os
import numpy as np
from tqdm import tqdm


class NpairDataset:

    def __init__(self, dataset, transform):
        # super(TripletDataset, self).__init__(path, transform)

        self.transform = transform

        self.data_set = dataset
        self.length = len(dataset.targets)
        print('Generating {} pairs'.format(self.length))
        self.classes = len(self.data_set.classes)
        # 这里生成N元组
        self.train_npair = self.generate_npair(self.data_set, self.length)

    def get_dataset(self, path):
        data_train = torchvision.datasets.CIFAR10(root=path, train=True, transform=self.transform)
        return data_train,len(data_train)

    @staticmethod
    def generate_npair(data_set, length):
        pairs = []

        for x in tqdm(range(length)):
            # 生成正负样本从0到classes-1的编号
            p_index = np.random.randint(0, len(data_set))
            while data_set.targets[x] != data_set.targets[p_index]:
                p_index = np.random.randint(0, len(data_set))

            # ps = data_set.targets[p_index]
            # ns = data_set.targets[n_index]
            pairs.append([data_set.data[x], data_set.data[p_index], data_set.targets[p_index]])
        return pairs

    def __getitem__(self, index):
        '''

        Args:
            index: Index of the triplet or the matches - not of a single image

        Returns:

        '''
        # Get the index of each image in the triplet
        img_a, img_p, p = self.train_npair[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_a = Image.fromarray(img_a)
        img_p = Image.fromarray(img_p)

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)

        return img_a, img_p, p

    def __len__(self):
        return len(self.train_npair)
