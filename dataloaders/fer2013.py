import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import h5py


class FER2013_Dataset(Dataset):
    def __init__(self, csv_file, img_dir, datatype, transform, h5_path=None):
        '''
        Pytorch Dataset class
        params:-
                 csv_file : the path of the csv file    (train, validation, test)
                 img_dir  : the directory of the images (train, validation, test)
                 datatype : string for searching along the image_dir (train, val, test)
                 transform: pytorch transformation over the data
                 h5_path  : h5 file to read from ({train, val, test}_{imgs, labels})
        return :-
                 image, labels
        '''
        self.h5_path = h5_path
        self.img_dir = img_dir
        self.transform = transform
        self.datatype = datatype
        if h5_path:
            print(f"Reading h5 file from: {h5_path}")
            with h5py.File(h5_path, 'r') as hf:
                self.imgs = hf[datatype+'_imgs'][:]
                self.labels = hf[datatype+'_labels'][:]
            print("READ SHAPE: ", len(self.imgs), len(self.labels))
        else:
            self.csv_file = pd.read_csv(csv_file)
            self.labels = self.csv_file['emotion']


    def show(self):
        idx = np.random.randint(0, len(self.csv_file))
        img = Image.open(self.img_dir+self.datatype+str(idx)+'.jpg')
        labels = torch.from_numpy(np.array(self.labels[idx])).long()
        img.show(title=f"{labels}")
        print("Label = ", labels)

    def __len__(self):
        if self.h5_path:
            return len(self.imgs)
        else:
            return len(self.csv_file)

    def __getitem__(self, idx):
        if self.h5_path:
            img = self.imgs[idx]
            label = self.labels[idx]

        else:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img = Image.open(self.img_dir+self.datatype+str(idx)+'.jpg')
            label = np.array(self.labels[idx])
            label = torch.from_numpy(label).long()

        if self.transform:
            img = self.transform(img)
        return img, label

#Helper function
def eval_data_dataloader(csv_file,img_dir,datatype,sample_number,transform = None):
    '''
    Helper function used to evaluate the Dataset class
    params:-
            csv_file : the path of the csv file    (train, validation, test)
            img_dir  : the directory of the images (train, validation, test)
            datatype : string for searching along the image_dir (train, val, test)
            sample_number : any number from the data to be shown
    '''
    if transform is None :
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    dataset = FER2013_Dataset(csv_file=csv_file,img_dir = img_dir,datatype = datatype,transform = transform)

    label = dataset.__getitem__(sample_number)[1]
    print(label)
    imgg = dataset.__getitem__(sample_number)[0]
    imgnumpy = imgg.numpy()
    imgt = imgnumpy.squeeze()
    plt.imshow(imgt)
    plt.show()



class Generate_data():
    def __init__(self, datapath, h5_path=None):
        """
        Generate_data class
        Two methods to be used
        1-split_data:
            - Split icml_face_data.csv into train.csv / val.csv / test.csv based on Usage column
        2-save_images:
            - Save train/val/test images into different directories
        [Note] that you have to split the public and private from fer2013 file
        """
        self.data_path = datapath
        self.split_data()
        self.save_images('train')
        if h5_path:
            with h5py.File(h5_path, "w") as hf:
                print(type(self.imgs), type(self.imgs[0]), type(self.labels), type(self.labels[0]))
                hf.create_dataset("train_imgs", data=self.imgs)
                hf.create_dataset("train_labels", data=self.labels)

        self.save_images('val')
        if h5_path:
            with h5py.File(h5_path, "a") as hf:
                hf.create_dataset("val_imgs", data=self.imgs)
                hf.create_dataset("val_labels", data=self.labels)

        self.save_images('test')
        if h5_path:
            with h5py.File(h5_path, "a") as hf:
                hf.create_dataset("test_imgs", data=self.imgs)
                hf.create_dataset("test_labels", data=self.labels)


    def split_data(self, test_filename = 'test', val_filename= 'val'):
        """
        Helper function to split the validation and test data from general test file as it contains (Public test, Private test)
            params:-
                data_path = path to the folder that contains the test data file
        """
        # csv_path = self.data_path +"/"+ 'test.csv'
        csv_path = self.data_path +"/"+ 'icml_face_data.csv'
        df = pd.read_csv(csv_path)
        # strip spaces from column names
        df = df.rename(columns=lambda x: x.strip())
        train = df.loc[df['Usage'] == 'Training']
        val = df.loc[df['Usage'] == 'PublicTest']
        test = df.loc[df['Usage'] == 'PrivateTest']

        train.to_csv(self.data_path+"/"+"train"+".csv")
        test.to_csv(self.data_path+"/"+test_filename+".csv")
        val.to_csv(self.data_path+"/"+val_filename+".csv")
        print("Done splitting the test file into validation & final test file")

    def str_to_image(self, str_img = ' '):
        '''
        Convert string pixels from the csv file into image object
            params:- take an image string
            return :- return PIL image object
        '''
        imgarray_str = str_img.split(' ')
        imgarray = np.asarray(imgarray_str,dtype=np.uint8).reshape(48,48)
        return Image.fromarray(imgarray)

    def save_images(self, datatype='train'):
        '''
        save_images is a function responsible for saving images from data files e.g(train, test) in a desired folder
            params:-
            datatype= str e.g (train, val, finaltest)
        '''
        # clear imgs/labels
        self.imgs = []
        self.labels = []

        foldername= self.data_path+"/"+datatype
        csvfile_path= self.data_path+"/"+datatype+'.csv'
        if not os.path.exists(foldername):
            os.mkdir(foldername)

        data = pd.read_csv(csvfile_path)
        images = data['pixels'] #dataframe to series pandas
        labels = data['emotion']
        numberofimages = images.shape[0]
        for index in tqdm(range(numberofimages)):
            img = self.str_to_image(images[index])
            label = labels[index]
            self.imgs.append(np.array(img))
            self.labels.append(label)
            img.save(os.path.join(foldername,'{}{}.jpg'.format(datatype,index)),'JPEG')
        print('Done saving {} data'.format((foldername)))



class FER2013_Dataloader:
    def __init__(self, data_dir="data/fer2013/", batchsize=128, num_workers=4, gen_data=False, h5_path=None):
        """
        generate train / val / test loaders
        FER2013: 48x48 grayscale images
        """
        if gen_data:
            Generate_data(data_dir, h5_path)
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)),
        ])

        train_csv_file = data_dir +'/train.csv'
        validation_csv_file = data_dir + '/val.csv'
        train_img_dir = data_dir + '/train/'
        validation_img_dir = data_dir + '/val/'

        train_ds = FER2013_Dataset(train_csv_file, train_img_dir, "train", self.transform, h5_path=h5_path)
        self.train_len = len(train_ds)
        self.train_loader = DataLoader(
            train_ds,
            batch_size = batchsize,
            shuffle = True,
            num_workers = num_workers
        )

        val_ds = FER2013_Dataset(validation_csv_file, validation_img_dir, "val", self.transform, h5_path=h5_path)
        self.val_len = len(val_ds)
        self.val_loader = DataLoader(
            val_ds,
            batch_size = batchsize,
            shuffle = False,
            num_workers = num_workers
        )


if __name__ == "__main__":
    # ONLY FOR TESTING
    dl = FER2013_Dataloader(gen_data=False, h5_path="./data/fer2013.h5")
    print(dl.train_len, dl.val_len)
    for data, label in dl.train_loader:
        print(data.shape, label)
