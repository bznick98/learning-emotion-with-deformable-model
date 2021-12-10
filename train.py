import argparse
import numpy as np
import gc
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, Subset
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

# utility tools
from utils.util_tools import choose_dataset, choose_model, train_epoch, val_epoch, device_setup, get_augmentations, plot
from datasets.map_dataset import MapDataset



# parsing args
parser = argparse.ArgumentParser(description="Configuration of setup and training process")
parser.add_argument('-s', '--setup', action='store_true', help='setup the dataset for the first time')
parser.add_argument('-d', '--data', type=str, default='/content/gdrive/MyDrive/CK_PLUS/',
                            help='Image folders for loading data, can be:\n\
                                FER2013: folder path that contains csv that downloaded from kaggle (icml_face_data.csv)\n\
                                FER_CKPLUS: folder path that contains 8 subfolders\n\
                                CK_PLUS: folder path that contains 7 subfolders')
parser.add_argument('-ds', '--dataset', type=str, default='CK_PLUS', help='choice of \{FER2013, FER_CKPLUS, CK_PLUS\}')

# model arch settings
parser.add_argument('-m', '--model', type=str, default='de', help='DL model to run, can only be one of {de, de224, vgg, simple}')
parser.add_argument('-lrsc', '--schedule', action='store_true', help='if enabled, lr will be scheduled using ReduceLROnPlateau')
parser.add_argument('-dc', '--de_conv', action='store_true', help='if enabled, replacing (some) conv with deformable conv')
parser.add_argument('-wide', '--wide', action='store_true', help='if enabled, deep_emotion will be wider, increase channel=>64')
parser.add_argument('-deep', '--deep', action='store_true', help='if enabled, deep_emotion will be deeper, add 2 more conv layers')

# hyperparameters
parser.add_argument('-k', '--k_fold', type=int, default=0, help= 'k-fold cross validation, if=0, do normal training')
parser.add_argument('-e', '--epochs', type=int, default=100, help= 'number of epochs')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='value of learning rate')
parser.add_argument('-bs', '--batch_size', type=int, default=128, help='training/validation batch size')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4, help='weight decay coeff(L2 regularization)')
parser.add_argument('-drop', '--dropout', type=float, default=0, help='dropout rate, 0-1, 0=no dropout')

# data augmentation options
parser.add_argument('-hflip', '--random_hflip', action='store_true', help='if enabled, add transforms.RandomHorizontalFlip() to aug')
parser.add_argument('-hflip_prob', '--random_hflip_prob', type=float, default=0.5, help='if -hflip enabled, set probability')
parser.add_argument('-rcrop', '--random_crop', action='store_true', help='if enabled, add transforms.RandomCrop() to aug')
parser.add_argument('-rcrop_size', '--random_crop_size', type=int, default=224, help='if -rcrop enabled, set size (if=224, then crop 224x224)')
parser.add_argument('-rjitter', '--random_jitter', action='store_true', help='if enabled, add transforms.RandomColorJitter() to aug')
parser.add_argument('-rjitter_b', '--random_jitter_brightness', type=float, default=0.3, help='if -rjitter enabled, set brightness, 0-1')
parser.add_argument('-resize', '--resize', type=int, default=-1, help='if specified (>0), resize to this size')


args = parser.parse_args()



def train_kfold(net, epochs, dataset, batch_size, lr, wd, k=10, input_size=(224,224), lr_schedule=False, augmentations=None):
    '''
    Training Loop
    '''
    # setting up device
    device = device_setup()

    net.to(device)

    # K-Folds split cross validation
    if k == 0:
        splits=KFold(n_splits=10,shuffle=True)
    else:
        splits=KFold(n_splits=k,shuffle=True)

    train_loss_kfolds = []
    val_loss_kfolds = []
    train_acc_kfolds = []
    val_acc_kfolds = []
    max_train_acc_kfolds = []
    max_val_acc_kfolds = []

    summary(net, input_size=(batch_size, 1, input_size[0], input_size[1]), verbose=1)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        if lr_schedule:
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=1, min_lr=1e-7)

        # Loss
        criterion = nn.CrossEntropyLoss()

        # print model info
        print(f"=================================== Start Training fold {fold+1}/{k+1} ===================================")
        # k-folds sampling
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # augment training set
        if augmentations:
            augmentations = transforms.Compose(augmentations)
            train_subset = MapDataset(train_subset, augmentations)
        # if there is resize, resize both training and validation data
        if args.resize > 0:
            train_subset = MapDataset(train_subset, [transforms.Resize((args.resize, args.resize))])
            val_subset = MapDataset(val_subset, [transforms.Resize((args.resize, args.resize))])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)
        train_len = len(train_subset)
        val_len = len(val_subset)

        train_loss_arr = []
        val_loss_arr = []
        train_acc_arr = []
        val_acc_arr = []

        max_train_acc = 0
        max_val_acc = 0

        for e in range(epochs):
            # train the model (train epoch)
            train_loss, train_acc = train_epoch(net, criterion, device, train_loader, train_len, optimizer)

            # validate the model (validate epoch)
            val_loss, val_acc = val_epoch(net, criterion, device, val_loader, val_len)

            # record the max train/val accuracy acchieved of all time
            max_train_acc = max(max_train_acc, train_acc)
            max_val_acc = max(max_val_acc, val_acc)

            # learning rate scheduler
            if lr_schedule:
              scheduler.step(val_loss)

            # epoch info
            print(f'Epoch: {e+1} \tTraining Loss: {train_loss:.8f} \tValidation Loss {val_loss:.8f} \tTraining Accuarcy {train_acc*100:.3f}% \tValidation Accuarcy {val_acc*100:.3f}%')

            # save for plotting
            train_loss_arr.append(train_loss)
            train_acc_arr.append(train_acc)
            val_loss_arr.append(val_loss)
            val_acc_arr.append(val_acc)
        

        # history for each fold
        train_loss_kfolds.append(train_loss_arr)
        val_loss_kfolds.append(val_loss_arr)
        train_acc_kfolds.append(train_acc_arr)
        val_acc_kfolds.append(val_acc_arr)
        max_train_acc_kfolds.append(max_train_acc)
        max_val_acc_kfolds.append(max_val_acc)

        torch.save(net.state_dict(),'deep_emotion-{}-{}.pt'.format(epochs, batch_size))
        # clean-up after all epoch
        torch.cuda.empty_cache()

        # normal training ends here (using 10-fold split, but run 1 time)
        if k == 0:
            break

    return train_loss_kfolds, train_acc_kfolds, val_loss_kfolds, val_acc_kfolds, max_train_acc_kfolds, max_val_acc_kfolds






if __name__ == "__main__":
    # choose dataset based on args config
    dataset = choose_dataset(args)
    img_size = dataset[0][0].detach().numpy().shape[1:]   # 2d image size (48x48) or (224x224)

    # augmentation will have final training image size => img_size
    augment_list, img_size = get_augmentations(args, input_size=img_size)

    # choose model based on args config
    net = choose_model(args, input_size=img_size)

    # k-fold cross validation
    history = train_kfold(net=net,
                            epochs=args.epochs,
                            dataset=dataset,
                            batch_size=args.batch_size,
                            lr=args.learning_rate,
                            wd=args.weight_decay,
                            k=args.k_fold,
                            input_size=img_size, 
                            lr_schedule=args.schedule,
                            augmentations=augment_list)

    print("=============== Current Running Configuration ===============")
    print(args)
    print("=============================================================")
    
    # plot
    plot(history)
    