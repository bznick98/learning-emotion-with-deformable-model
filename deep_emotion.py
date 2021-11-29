import torch
import torch.nn as nn
import torch.nn.functional as F

from dcn import DeformableConv2d

class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        input: Nx48x48x1
        '''
        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)      # 46x46x32
        self.bn1 = nn.BatchNorm2d(10)     # 18x18x128
        self.conv2 = nn.Conv2d(10,10,3)     # 44x44x64
        self.bn2 = nn.BatchNorm2d(10)     # 18x18x128
        self.pool2 = nn.MaxPool2d(2,2)      # 22x22x64

        # replace conv3/4 with deformable convolution
        self.de_conv3 = DeformableConv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=0)
        self.de_conv4 = DeformableConv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=0)

        self.conv3 = nn.Conv2d(10,10,3)    # 20x20x128
        self.bn3 = nn.BatchNorm2d(10) 
        self.conv4 = nn.Conv2d(10,10,3)   # 18x18x128
        self.bn4 = nn.BatchNorm2d(10)  
        self.pool4 = nn.MaxPool2d(2,2)

        # self.bn2 = nn.BatchNorm2d(128)     # 18x18x128

        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(9 * 9 * 128, 50)        # 
        self.fc1 = nn.Linear(810, 50)        # 
        self.bn5 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,7)

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        # xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,input):
        out = self.stn(input)

        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = self.bn1(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.bn2(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        # out = out.view(-1, 1327104)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out
