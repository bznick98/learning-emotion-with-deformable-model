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


class FER_CKPLUS_Dataset(Dataset):
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
            for file in tqdm(os.listdir(img_dir + "/" + dir), desc=f"Loading {dir}"):
                with Image.open(img_dir + "/" + dir + "/" + file) as img:
                    self.imgs.append(np.array(img))
                    self.labels.append(curr_label)

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

        

class FER_CKPLUS_Dataloader:
    def __init__(self, data_dir="data/fer_ckplus_kdef/", batchsize=128, num_workers=4, resize=(128,128)):
        """
        generate train loader
        """
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)),
            transforms.Resize(resize)
        ])
        
        train_ds = FER_CKPLUS_Dataset(data_dir, self.transform)
        self.train_len = len(train_ds)
        self.train_loader = DataLoader(
            train_ds,
            batch_size = batchsize,
            shuffle = True,
            num_workers = num_workers
        )

        self.val_len = 0
        self.val_loader = None





if __name__ == "__main__":
    test_dataset = FER_CKPLUS_Dataset("data/fer_ckplus_kdef")
    test_loader = DataLoader(test_dataset,batch_size=128,shuffle = True,num_workers=0)