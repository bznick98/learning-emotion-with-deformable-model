import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple_CNN(nn.Module):
    """
    Simple CNN, to test if simple model still overfits data
    """
    def __init__(self, drop=0, n_drop=1):
        '''
        input: Nx1x48x48
            - drop: dropout rate before 1st FC layer
        '''
        super().__init__()
        self.n_drop = n_drop

        self.conv1 = nn.Conv2d(1,32,3,padding='same')       # 48x48
        self.bn1 = nn.BatchNorm2d(32)     
        self.pool1 = nn.MaxPool2d(2,2)                      # 24x24

        self.conv2 = nn.Conv2d(32,32,3,padding='same')      # 24x24
        self.bn2 = nn.BatchNorm2d(32)     
        self.pool2 = nn.MaxPool2d(2,2)                      # 12x12

        self.conv3 = nn.Conv2d(32,32,3)                     # 10x10
        self.bn3 = nn.BatchNorm2d(32) 
        self.pool3 = nn.MaxPool2d(2,2)                      # 5x5x32

        self.flatten = nn.Flatten()
        # handle 48x48 input
        self.fc1 = nn.Linear(5 * 5 * 32, 32)
        self.dropout = nn.Dropout(drop)
        self.bn_fc = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32,7)

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

    def forward(self,input):
        # 1st conv
        out = F.relu(self.conv1(input))
        out = self.bn1(out)
        out = self.pool1(out)

        # 2nd conv
        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        out = self.pool2(out)
        
        # 3rd conv
        out = F.relu(self.conv3(out))
        out = self.bn3(out)
        out = self.pool3(out)

        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = self.bn_fc(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if self.n_drop == 2:
            out = self.dropout(out)

        return out

