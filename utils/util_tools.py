"""
Utility tools used for training
"""
import torch
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# datasets
from datasets.fer_ckplus_kdef import FER_CKPLUS_Dataset
from datasets.fer2013 import FER2013_Dataset, Generate_data

# models
from models.deep_emotion import Deep_Emotion, Deep_Emotion224
from models.simple import Simple_CNN
from models.vgg import Vgg


def choose_dataset(args):
    """
    choosing the correct dataset
    """
    if args.dataset == "FER2013": # FER2013
        if args.data:
            read_dir = args.data
        else:
            read_dir = "/content/gdrive/MyDrive/FER_2013/" # read from google drive
        if args.setup:
            Generate_data()
        else:
            train_csv_file = os.path.join(read_dir, '/train.csv')
            val_csv_file = os.path.join(read_dir, '/val.csv')
            test_csv_file = os.path.join(read_dir, '/test.csv')

            train_img_dir = os.path.join(read_dir, '/train/')
            val_img_dir = os.path.join(read_dir, '/val/')
            test_img_dir = os.path.join(read_dir, '/test/')

            h5_path = os.path.join(read_dir, "fer2013.h5")

            train_ds = FER2013_Dataset(csv_file=train_csv_file,
                                    img_dir=train_img_dir,
                                    datatype="train",
                                    h5_path=h5_path)
            val_ds = FER2013_Dataset(csv_file=val_csv_file,
                                    img_dir=val_img_dir,
                                    datatype="val",
                                    h5_path=h5_path)
            test_ds = FER2013_Dataset(csv_file=test_csv_file,
                                    img_dir=test_img_dir,
                                    datatype="test",
                                    h5_path=h5_path)
            return train_ds, val_ds, test_ds


    elif args.dataset == "FER_CKPLUS": # FER_CKPLUS_KDEF
        if args.data:
            read_dir = args.data
        else:
            read_dir = "/content/gdrive/MyDrive/FER_CKPLUS/" # read from google drive
        ds = FER_CKPLUS_Dataset(img_dir=read_dir, h5_path=os.path.join(read_dir, "fer_ckplus.h5"))
                                
    elif args.dataset == "CK_PLUS": # CK+ 
        if args.data:
            read_dir = args.data
        else:
            read_dir = "/content/gdrive/MyDrive/CK_PLUS/" # read from google drive
        ds = FER_CKPLUS_Dataset(img_dir=read_dir, h5_path=os.path.join(read_dir, "CK_PLUS.h5"))

    else:
        raise Exception("-d or --data can only be \{FER2013, FER_CKPLUS, CK_PLUS\}")
    
    return ds


def choose_model(args, input_size=(224,224)):
    # TODO: use input size information in building mode instead of passing hardcoded flag
    input_224 = False
    if input_size[0] == 224:
        input_224 = True

    # Choosing model
    if args.model == 'de':
        net = Deep_Emotion(wider=args.wide, deeper=args.deep, de_conv=args.de_conv, input_224=input_224, drop=args.dropout, n_drop=args.num_dropout)
    elif args.model == 'vgg':
        net = Vgg()
    elif args.model == 'de224':
        net = Deep_Emotion224(de_conv=args.de_conv, drop=args.dropout, n_drop=args.num_dropout)
    elif args.model == 'simple':
        net = Simple_CNN(drop=args.dropout, n_drop=args.num_dropout)
    else:
        raise Exception("-m or --model can only accpet one of \{de, vgg, de224, simple\}")

    return net


def get_augmentations(args, input_size):
    """
    specify augmentation, out_size specified final return image size
    input:
        - args
        - input_size: raw input image size loaded from data
    return:
        - aug_list: list of transforms.XXX
        - out_size: final image size for training
    """
    out_size = input_size
    if args.random_crop:
        out_size = (args.random_crop_size, args.random_crop_size)
    if args.resize > 0:
        out_size = (args.resize, args.resize)

    aug_list = []
    if args.random_hflip:
        aug_list.append(transforms.RandomHorizontalFlip(args.random_hflip_prob))
    if args.random_crop:
        aug_list.append(transforms.RandomCrop(args.random_crop_size))

    # if specified resize size, otherwise resize will be loaded image size    
    return aug_list, out_size


def train_epoch(net, criterion, device, train_loader, train_len, optimizer):
    """
    train data for 1 epoch
    """
    net.train()
    train_loss = 0
    train_correct = 0
    
    with tqdm(train_loader, unit="batch") as tepoch:
        for data, labels in tepoch:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs,1)
            train_correct += float(torch.sum(preds == labels.data))
    
    train_acc = train_correct / train_len
    train_loss =  train_loss / train_len
    return train_loss, train_acc


def val_epoch(net, criterion, device, val_loader, val_len):
    """
    Validate data for 1 epoch
    """
    val_loss = 0
    val_correct = 0

    net.eval()
    with torch.no_grad():
        for data,labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            val_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs,1)
            val_correct += float(torch.sum(val_preds == labels.data))

    val_loss =  val_loss / val_len
    val_acc = val_correct / val_len

    return val_loss, val_acc


def device_setup():
    """
    return torch device type
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    print()
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
    return device


def plot(history, save=None):
    # Plotting loss/acc
    import matplotlib.pyplot as plt

    train_loss_kfolds, train_acc_kfolds, val_loss_kfolds, val_acc_kfolds, max_train_acc_kfolds, max_val_acc_kfolds = history # returned from train()
    epochs = len(train_loss_kfolds[0])
    xs = [i for i in range(epochs)]

    fig, axes = plt.subplots(1,4, figsize=(20,4))
    for train_loss_arr in train_loss_kfolds:
        axes[0].plot(xs, train_loss_arr)
    axes[0].set_title("Train Loss vs epochs")

    for train_acc_arr in train_acc_kfolds:
        axes[1].plot(xs, train_acc_arr)
    axes[1].set_title("Train Accuracy vs epochs")

    for val_loss_arr in val_loss_kfolds:
        axes[2].plot(xs, val_loss_arr)
    axes[2].set_title("Val Loss vs epochs")

    for val_acc_arr in val_acc_kfolds:
        axes[3].plot(xs, val_acc_arr)
    axes[3].set_title("Val Accuracy vs epochs")

    # Average Acc for K-folds from LAST epoch
    # avg_train_acc = sum([arr[-1] for arr in train_acc_kfolds]) / len(train_acc_kfolds)
    # avg_val_acc = sum([arr[-1] for arr in val_acc_kfolds]) / len(val_acc_kfolds)

    # Average Acc for K-folds from HIGHEST epoch
    avg_train_acc = np.mean(max_train_acc_kfolds)
    avg_val_acc = np.mean(max_val_acc_kfolds)

    print(f"{len(train_acc_kfolds)}-fold average Training Acc = {avg_train_acc}")
    print(f"{len(val_acc_kfolds)}-fold average Validation Acc = {avg_val_acc}")

    if save:
        filename = f'acc-loss-e{epochs}.png'
        fig.savefig(filename)
        print(f"Plot saved at {os.path.abspath(filename)}")
    plt.show()



def print_config(args):
    """
    print :)
    """
    print("=============== Current Running Configuration ===============")
    print(args)
    print("=============================================================")
