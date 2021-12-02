from argparse import ArgumentParser
from typing import *
import timeit
import pytorch_lightning as pl
from torch.utils.data import random_split, Dataset,DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import pickle
import os,torch,random
import torchnet as tnt
from PIL import Image as pil_image


def data2datalabel(ori_data):
    data = []
    label = []
    for c_idx in ori_data:
        for i_idx in range(len(ori_data[c_idx])):
            data.append(ori_data[c_idx][i_idx])
            label.append(c_idx)
    return data, label


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds


class Cifar(Dataset):
    """
    preprocess the MiniImageNet dataset
    """

    def __init__(self, root, partition='train', category='cifar'):
        super(Cifar, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 32, 32]
        # set normalizer
        mean_pix = [x/255.0 for x in [129.37731888,
                                      124.10583864, 112.47758569]]
        std_pix = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=.1, contrast=.1, saturation=.1, hue=.1),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])
        print('Loading {} dataset -phase {}'.format(category, partition))
        # load data
        if category == 'cifar':
            dataset_path = os.path.join(
                self.root, 'cifar-fs', 'cifar_fs_%s.pickle' % self.partition)
            with open(dataset_path, 'rb') as handle:
                u = pickle._Unpickler(handle)
                u.encoding = 'latin1'
                data = u.load()
            self.data = data['data']
            self.labels = data['labels']
            self.label2ind = buildLabelIndex(self.labels)
            self.full_class_list = sorted(self.label2ind.keys())
        else:
            print('No such category dataset')

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        image_data = pil_image.fromarray(img)
        return image_data, label

    def __len__(self):
        return len(self.data)


class MiniImagenet(Dataset):
    """
    preprocess the MiniImageNet dataset
    """

    def __init__(self, root, partition='train', category='mini'):
        super(MiniImagenet, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]
        # set normalizer
        mean_pix = [x / 255.0 for x in [120.39586422,
                                        115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=.1,
                                       contrast=.1,
                                       saturation=.1,
                                       hue=.1),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        print('Loading {} ImageNet dataset -phase {}'.format(category, partition))
        # load data
        dataset_path = os.path.join(
            self.root, 'mini_imagenet', 'mini_imagenet_%s.pickle' % self.partition)
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)

        self.full_class_list = list(data.keys())
        self.data, self.labels = data2datalabel(data)
        self.label2ind = buildLabelIndex(self.labels)

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        image_data = pil_image.fromarray(np.uint8(img))
        image_data = image_data.resize((self.data_size[2], self.data_size[1]))
        return image_data, label

    def __len__(self):
        return len(self.data)


class TieredImagenet(Dataset):
    def __init__(self, root, partition='train', category='tiered'):
        super(TieredImagenet, self).__init__()

        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]

        # set normalizer
        mean_pix = [x/255.0 for x in [120.39586422,
                                      115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272, 68.27635443,  72.54505529]]

        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=.1, contrast=.1, saturation=.1, hue=.1),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])
        print('Loading {} ImageNet dataset -phase {}'.format(category, partition))
        if category == 'tiered':
            dataset_path = os.path.join(
                self.root, 'tiered-imagenet', '%s_images.npz' % self.partition)
            label_path = os.path.join(
                self.root, 'tiered-imagenet', '%s_labels.pkl' % self.partition)
            with open(dataset_path, 'rb') as handle:
                self.data = np.load(handle)['images']
            with open(label_path, 'rb') as handle:
                label_ = pickle.load(handle)
                self.labels = label_['labels']
                self.label2ind = buildLabelIndex(self.labels)
            self.full_class_list = sorted(self.label2ind.keys())
        else:
            print('No such category dataset')

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        image_data = pil_image.fromarray(img)
        return image_data, label

    def __len__(self):
        return len(self.data)


class CUB200(Dataset):
    def __init__(self, root, partition='train', category='cub'):
        super(CUB200, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]
        # set normalizer
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(84, interpolation=pil_image.BICUBIC),
                transforms.RandomCrop(84, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=.1, contrast=.1, saturation=.1, hue=.1),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([
                transforms.Resize(84, interpolation=pil_image.BICUBIC),
                transforms.CenterCrop(84),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])
        print('Loading {} dataset -phase {}'.format(category, partition))
        if category == 'cub':
            IMAGE_PATH = os.path.join(self.root, 'cub-200-2011', 'images')
            txt_path = os.path.join(
                self.root, 'cub-200-2011/split', '%s.csv' % self.partition)
            lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
            data = []
            label = []
            lb = -1
            self.wnids = []
            for l in lines:
                context = l.split(',')
                name = context[0]
                wnid = context[1]
                path = os.path.join(IMAGE_PATH, wnid, name)
                if wnid not in self.wnids:
                    self.wnids.append(wnid)
                    lb += 1
                data.append(path)
                label.append(lb)
            self.data = data
            self.labels = label
            self.full_class_list = list(np.unique(np.array(label)))
            self.label2ind = buildLabelIndex(self.labels)
        else:
            print('No such category dataset')

    def __getitem__(self, index):
        path, label = self.data[index], self.labels[index]
        image_data = pil_image.open(path).convert('RGB')
        return image_data, label

    def __len__(self):
        return len(self.data)

# TODO: 用torch的dataloader实现
class DataLoader:
    """
    The dataloader of DPGN model for MiniImagenet dataset
    """
    def __init__(self, dataset, num_tasks, num_ways, num_shots, num_queries, epoch_size, num_workers=4, batch_size=1):

        self.dataset = dataset
        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.data_size = dataset.data_size
        self.full_class_list = dataset.full_class_list
        self.label2ind = dataset.label2ind
        self.transform = dataset.transform
        self.phase = dataset.partition
        self.is_eval_mode = (self.phase == 'test') or (self.phase == 'val')

        
    def get_task_batch(self):
        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(self.num_ways * self.num_shots):
            data = np.zeros(shape=[self.num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[self.num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(self.num_ways * self.num_queries):
            data = np.zeros(shape=[self.num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[self.num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)
        # for each task
        for t_idx in range(self.num_tasks):
            task_class_list = random.sample(self.full_class_list, self.num_ways)
            # for each sampled class in task
            for c_idx in range(self.num_ways):
                data_idx = random.sample(self.label2ind[task_class_list[c_idx]], self.num_shots + self.num_queries)
                class_data_list = [self.dataset[img_idx][0] for img_idx in data_idx]
                for i_idx in range(self.num_shots):
                    # set data
                    support_data[i_idx + c_idx * self.num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * self.num_shots][t_idx] = c_idx
                # load sample for query set
                for i_idx in range(self.num_queries):
                    query_data[i_idx + c_idx * self.num_queries][t_idx] = \
                        self.transform(class_data_list[self.num_shots + i_idx])
                    query_label[i_idx + c_idx * self.num_queries][t_idx] = c_idx
        support_data = torch.stack([torch.from_numpy(data).float() for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float() for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float() for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float() for label in query_label], 1)
        return support_data, support_label, query_data, query_label

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            support_data, support_label, query_data, query_label = self.get_task_batch()
            return support_data, support_label, query_data, query_label

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(1 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))
        return data_loader
    
    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size // self.batch_size
     
    def __iter__(self):
        return self.get_iterator(0)


class DatasetModule(pl.LightningDataModule):
    def __init__(self, data_name, 
                train_opt,
                 data_dir: str = "/data",
                 num_workers = 4,
                 train_bs=256,
                 val_bs=256,
                 test_bs=256,**kwargs):
        super().__init__()
        self.train_opt = train_opt
        self.data_dir = data_dir
        self.data_name = data_name
        self.train_bs = train_bs
        self.val_bs = val_bs
        self.test_bs = test_bs
        self.num_workers = num_workers 

        if data_name =='cifar':
            data = (Cifar(data_dir,par,data_name) for par in ('train','val','test'))
        elif data_name =='mini':
            data = (MiniImagenet(data_dir,par,data_name) for par in ('train','val','test'))
        else:
            NotImplemented()
        self.train_data, self.val_data, self.test_data = data

    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group('dataset')
        parser.add_argument('--data_dir',default='/data')
        parser.add_argument('--data_name', default='cifar', type=str)
        parser.add_argument('--train_bs', default=256)
        parser.add_argument('--val_bs', default=256)
        parser.add_argument('--test_bs', default=256)
        return parser


    def train_dataloader(self):
        train_opt = self.train_opt
        return DataLoader(self.train_data,
                              num_tasks=train_opt['batch_size'],
                              num_ways=train_opt['num_ways'],
                              num_shots=train_opt['num_shots'],
                              num_queries=train_opt['num_queries'],
                              epoch_size=train_opt['iteration'])

    def val_dataloader(self):
        train_opt = self.train_opt
        return DataLoader(self.val_data,
                              num_tasks=train_opt['batch_size'],
                              num_ways=train_opt['num_ways'],
                              num_shots=train_opt['num_shots'],
                              num_queries=train_opt['num_queries'],
                              epoch_size=train_opt['iteration'])

    def test_dataloader(self):
        train_opt = self.train_opt
        return DataLoader(self.test_data,
                              num_tasks=train_opt['batch_size'],
                              num_ways=train_opt['num_ways'],
                              num_shots=train_opt['num_shots'],
                              num_queries=train_opt['num_queries'],
                              epoch_size=train_opt['iteration'])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    DatasetModule.add_argparse_args(parser)
    args = parser.parse_args()
    train_opt = OrderedDict()
    train_opt['num_ways'] = 5
    train_opt['num_shots'] = 1
    train_opt['batch_size'] = 256
    train_opt['iteration'] = 100000
    train_opt['lr'] = 1e-3
    train_opt['weight_decay'] = 1e-5
    train_opt['dec_lr'] = 15000
    train_opt['dropout'] = 0.1
    train_opt['lr_adj_base'] = 0.1
    train_opt['loss_indicator'] = [1, 1, 0]
    train_opt['num_queries'] = 1
    if 'data_name' in args:
        data = DatasetModule(train_opt=train_opt,**args.__dict__)
        
        for i,b in enumerate(data.test_dataloader()): print(i)