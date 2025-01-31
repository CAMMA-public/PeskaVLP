import os
from fvcore.common.file_io import PathManager
from PIL import Image
import pandas as pd
from codes.registry import DATASETS
from torch.utils.data import Dataset
import pickle
import logging
from torch.utils.data import Dataset, ConcatDataset
import random
import numpy as np
from torchvision import transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

@DATASETS.register_module(name='Recognition_frame_bypass')
class BypassDataset(Dataset):
    def __init__(self, img_list, label_list, transforms=None, loader=pil_loader):
        self.img_list = img_list
        self.label_list = label_list

        assert len(self.img_list) == len(self.label_list)
        self.transform = transforms
        self.loader = loader


    def __getitem__(self, index):
        img_name = self.img_list[index]
        img = self.loader(img_name)

        label_phase = self.label_list[index]

        # print(imgs.size)
        if self.transform is not None:
            img = self.transform(img)

        final_dict = {'video':img, 'label': label_phase}
        return final_dict

    def __len__(self):
        return len(self.img_list)