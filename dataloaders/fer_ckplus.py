from __future__ import print_function
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class Plain_Dataset(Dataset):
    def __init__(self, img_dir, transform=None, **kargs):
        '''
        Pytorch Dataset class
        params:-
                 img_dir  : the directory of the images (root image dir)
                 transform: pytorch transformation over the data
        return :-
                 image, labels
        '''
        self.transform = transform
        self.label_names = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "sadness": 4,
            "surprise": 5,
            "neutrality": 6
        }
        self.imgs = []
        self.labels = []
        # read all subdirectories
        dirs = os.listdir(img_dir)
        for dir in dirs:
            if dir not in self.label_names: continue
            curr_label = self.label_names[dir]
            for file in tqdm(os.listdir(img_dir + "/" + dir)):
                with Image.open(img_dir + "/" + dir + "/" + file) as img:
                    self.imgs.append(np.array(img))
                    self.labels.append(curr_label)
                
        print("LOADED!")
        print(len(self.imgs))
        print(len(self.labels))

    def show(self):
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img ,label


if __name__ == "__main__":
    test_dataset = Plain_Dataset("data/fer_ckplus_kdef")
    test_loader = DataLoader(test_dataset,batch_size=128,shuffle = True,num_workers=0)