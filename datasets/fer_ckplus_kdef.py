import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import h5py



class FER_CKPLUS_Dataset(Dataset):
    def __init__(self, img_dir, transform=None, h5_path=None, **kargs):
        '''
        Pytorch Dataset class
        params:-
                 img_dir  : the directory of the images (root image dir)
                 transform: pytorch transformation over the data
                 h5_path  : used for saving / reading dataset from h5 file
        return :-
                 image, labels
        '''
        # base transform, transforms passed by parameter will be append to the end of list
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)),
        ]
        if transform:
            transform_list.extend(transform)
        
        self.transform = transforms.Compose(transform_list)

        self.label_names = {
            # fer_ck_plus_kdef & CK_PLUS_256
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "sadness": 4,
            "surprise": 5,
            "neutrality": 6,
            # CK_PLUS (48x48) doesn't seem to find neutral in these data
            "happy" : 3,
            "contempt": 6,
        }

        self.num_label_names = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6
        }
        # either read from h5 file or read from image subdirectories
        if h5_path:
            self.read_h5(filepath=h5_path)

        else:
            self.imgs = []
            self.labels = []
            # read all subdirectories
            dirs = os.listdir(img_dir)
            for dir in dirs:
                print(f"Reading from directory: {dir}")
                if dir in self.label_names:
                    curr_label = self.label_names[dir]
                elif dir in self.num_label_names:
                    curr_label = self.num_label_names[dir]
                else:
                    continue
                for file in tqdm(os.listdir(img_dir + "/" + dir), desc=f"Loading {dir}"):
                    with Image.open(img_dir + "/" + dir + "/" + file) as img:
                        self.imgs.append(np.array(img))
                        self.labels.append(curr_label)

    def show(self):
        pass

    def save_h5(self, savepath="./data/fer_ckplus.h5"):
        """
        save train/val dataset as single h5 file
        """
        print(f"Saving h5 file to: {savepath}")
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset("imgs", data=self.imgs)
            hf.create_dataset("labels", data=self.labels)

    def read_h5(self, filepath="./data/fer_ckplus.h5"):
        """
        read h5 file into self.imgs and self.labels
        """
        print(f"Reading h5 file from: {filepath}")
        with h5py.File(filepath, 'r') as hf:
            self.imgs = hf['imgs'][:]
            self.labels = hf['labels'][:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        img = self.transform(img)
        return img ,label

        

# class FER_CKPLUS_Dataloader:
#     def __init__(self, data_dir="data/fer_ckplus_kdef/", batchsize=128, num_workers=4, resize=None, augment=True, h5_path=None, train_val_split=0.9, transform=None):
#         """
#         generate train loader
#         """            
#         ds = FER_CKPLUS_Dataset(data_dir, transform=transform, resize=resize , h5_path=h5_path)
#         train_num = int(len(ds)*train_val_split) # default=0.9
#         val_num = len(ds) - train_num
#         train_ds, val_ds = random_split(ds, [train_num, val_num])
#         # not augment validation set
#         val_ds.is_train = False

#         # train loader
#         self.train_len = len(train_ds)
#         self.train_loader = DataLoader(
#             train_ds,
#             batch_size = batchsize,
#             shuffle = True,
#             num_workers = num_workers
#         )

#         # validation loader
#         self.val_len = len(val_ds)
#         self.val_loader = DataLoader(
#             val_ds,
#             batch_size = batchsize,
#             shuffle = True,
#             num_workers = num_workers
#         )






if __name__ == "__main__":
    dl = FER_CKPLUS_Dataloader("data/CK_PLUS")
    # test_loader = DataLoader(test_dataset,batch_size=128,shuffle = True,num_workers=0, h5_path="./data/fer_ckplus.h5")
    dataset = FER_CKPLUS_Dataset("data/CK_PLUS")
    dataset.save_h5(save_dir="./data/", data_name="CK_PLUS")
