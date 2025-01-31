#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:38:36 2021

@author: nwoye
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from codes.registry import DATASETS

@DATASETS.register_module()
class CholecT50():
    def __init__(self, dataset_dir, list_video, aug='original', split='train', preprocess=None):
        """ Args
                dataset_dir : common path to the dataset (excluding videos, output)
                list_video  : list video IDs, e.g:  ['VID01', 'VID02']
                aug         : data augumentation style
                split       : data split ['train', 'val', 'test']
            Call
                batch_size: int, 
                shuffle: True or False
            Return
                tuple ((image), (tool_label, verb_label, target_label, triplet_label))
        """
        if split=='test' or split=='val':
            video = list_video
            self.dataset = T50(img_dir = os.path.join(dataset_dir, 'data', video), 
                          triplet_file = os.path.join(dataset_dir, 'triplet', '{}.txt'.format(video)), 
                          tool_file = os.path.join(dataset_dir, 'instrument', '{}.txt'.format(video)),  
                          verb_file = os.path.join(dataset_dir, 'verb', '{}.txt'.format(video)),  
                          target_file = os.path.join(dataset_dir, 'target', '{}.txt'.format(video)), 
                          transform=preprocess)
            self.test_sampler = None# torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            iterable_dataset = []
            for video in list_video:
                dataset = T50(img_dir = os.path.join(dataset_dir, 'data', video), 
                            triplet_file = os.path.join(dataset_dir, 'triplet', '{}.txt'.format(video)), 
                            tool_file = os.path.join(dataset_dir, 'instrument', '{}.txt'.format(video)),  
                            verb_file = os.path.join(dataset_dir, 'verb', '{}.txt'.format(video)),  
                            target_file = os.path.join(dataset_dir, 'target', '{}.txt'.format(video)), 
                            transform=preprocess)
                iterable_dataset.append(dataset)
            self.dataset = ConcatDataset(iterable_dataset)
            self.test_sampler = None #torch.utils.data.distributed.DistributedSampler(self.dataset)

        
    def __call__(self, batch_size=25, shuffle=False):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=3, pin_memory=True, prefetch_factor=3*batch_size, persistent_workers=True, drop_last=True, sampler=self.test_sampler)
            
    
class T50(Dataset):
    def __init__(self, img_dir, triplet_file, tool_file, verb_file, target_file, transform=None):
        self.triplet_labels = np.loadtxt(triplet_file, dtype=np.int32, delimiter=',',)
        self.tool_labels = np.loadtxt(tool_file, dtype=np.int32, delimiter=',',)
        self.verb_labels = np.loadtxt(verb_file, dtype=np.int32, delimiter=',',)
        self.target_labels = np.loadtxt(target_file, dtype=np.int32, delimiter=',',)
        self.img_dir = img_dir
        self.transforms = transform
        self.size = 224
        self.num_frames = 16

        
    def __len__(self):
        return len(self.triplet_labels)
    
    def __getitem__(self, index):
        triplet_label = self.triplet_labels[index, 1:]
        tool_label = self.tool_labels[index, 1:]
        verb_label = self.verb_labels[index, 1:]
        target_label = self.target_labels[index, 1:]
        basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(6))
        img_path = os.path.join(self.img_dir, basename)
        image    = Image.open(img_path)

        image.convert('RGB')
        width = image.width
        height = image.height
        crop_size = min(width, height)
        new_height = crop_size
        new_width = crop_size

        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        # Crop the center of the image
        image = image.crop((left, top, right, bottom))
        # Resize the Image
        image = image.resize((self.size, self.size))
        # To tensor
        image = self.transforms(image)

        # video = image.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        # import torchvision.utils as utils
        # utils.save_image(video[:,0,...], 'tmp.png', normalize=True)
        # exit()
        return image, (tool_label, verb_label, target_label, triplet_label)


if __name__ == '__main__':
    dataset_dir = '/home2020/home/icube/kunyuan/cholectriplet/CholecT50-challenge-train'
    list_video = ['VID01', 'VID01']
    train_dataloader = CholecT50(dataset_dir, list_video)(batch_size=24, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels[0].size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    print(label)