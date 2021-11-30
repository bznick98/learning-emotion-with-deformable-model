import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# data
from dataloaders.fer_ckplus_kdef import FER_CKPLUS_Dataloader
from dataloaders.fer2013 import FER2013_Dataloader

# model
from models.deep_emotion import Deep_Emotion
from models.vgg import Vgg


# parsing args
parser = argparse.ArgumentParser(description="Configuration of setup and training process")
parser.add_argument('-s', '--setup', action='store_true', help='setup the dataset for the first time')
parser.add_argument('-d', '--data', type=str, default='data/',
                            help='data folder that contains data files that downloaded from kaggle (icml_face_data.csv)')
parser.add_argument('--model', type=str, default='de', help='DL model to run, can be one of {de, vgg}')
parser.add_argument('-e', '--epochs', type=int, default=100, help= 'number of epochs')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='value of learning rate')
parser.add_argument('-bs', '--batch_size', type=int, default=128, help='training/validation batch size')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4, help='weight decay coeff(L2 regularization)')
parser.add_argument('--show', action='store_true', help='Show 1 training sample.')
args = parser.parse_args()  

# cpu/gpu device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
print()
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


def Train(epochs, train_loader, val_loader, criterion, optmizer, device):
    '''
    Training Loop
    '''
    print("===================================Start Training===================================")
    for e in range(epochs):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0
        # Train the model  #
        net.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, labels in tepoch:
                data, labels = data.to(device), labels.to(device)
                optmizer.zero_grad()
                outputs = net(data)
                loss = criterion(outputs,labels)
                loss.backward()
                optmizer.step()
                train_loss += loss.item()
                _, preds = torch.max(outputs,1)
                train_correct += torch.sum(preds == labels.data)

        #validate the model#
        net.eval()
        for data,labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs,1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss/ dl.train_len
        train_acc = train_correct.double() / dl.train_len
        if val_loader:
          validation_loss =  validation_loss / dl.val_len
          val_acc = val_correct.double() / dl.val_len
        else:
          validation_loss = "No val"
          val_acc = 0
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}%'
                                                           .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))

    torch.save(net.state_dict(),'deep_emotion-{}-{}-{}.pt'.format(epochs,batchsize,lr))
    print("===================================Training Finished===================================")


if __name__ == '__main__':    
    # Choosing hyper-parameters  
    epochs = args.epochs
    lr = args.learning_rate
    batchsize = args.batch_size
    weight_decay = args.weight_decay

    # Choosing model
    if args.model == 'de':
        net = Deep_Emotion(wider=False, deeper=False, de_conv=False)
    elif args.model == 'vgg':
        net = Vgg()
    else:
        raise Exception("--model can only accpet one of \{de, vgg\}")
    net.to(device)

    # Choosing dataset
    if args.data == "FER2013": # FER2013
        summary(net, input_size=(batchsize, 1, 48, 48))
        dl = FER2013_Dataloader(gen_data=args.setup)
        train_loader = dl.train_loader
        val_loader = dl.val_loader
    elif args.data == "FER_CKPLUS": # FER_CKPLUS
        dl = FER_CKPLUS_Dataloader(resize=(48,48))
        train_loader = dl.train_loader
        val_loader = None
        summary(net, input_size=(batchsize, 1, 48, 48))
    else:
        raise Exception("-d or --data can only be \{FER2013, FER_CK\}")

    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optmizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    Train(epochs, train_loader, val_loader, criterion, optmizer, device)
