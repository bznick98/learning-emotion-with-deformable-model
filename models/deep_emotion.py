import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dcn import DeformableConv2d

class Deep_Emotion(nn.Module):
    def __init__(self, wider=False, deeper=False, de_conv=False):
        '''
        Deep_Emotion (wider)
        input: Nx1x48x48
        '''
        super(Deep_Emotion,self).__init__()
        self.wider = wider
        self.deeper = deeper
        self.de_conv = de_conv
        if wider:
            ch = 64
        else:
            ch = 10

        self.conv1 = nn.Conv2d(1,ch,3)      # 46x46xch
        self.bn1 = nn.BatchNorm2d(ch)     
        self.conv2 = nn.Conv2d(ch,ch,3)     # 44x44xch
        self.bn2 = nn.BatchNorm2d(ch)     
        self.pool2 = nn.MaxPool2d(2,2)      # 22x22xch

        # replace conv3/4 with deformable convolution
        if de_conv:
            self.de_conv3 = DeformableConv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=0)
            self.de_conv4 = DeformableConv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=0)

        self.conv3 = nn.Conv2d(ch,ch,3)    # 20x20xch
        self.bn3 = nn.BatchNorm2d(ch) 
        self.conv4 = nn.Conv2d(ch,ch,3)    # 18x18xch
        self.bn4 = nn.BatchNorm2d(ch)  
        self.pool4 = nn.MaxPool2d(2,2)     # 9x9xch

        # deeper network
        if deeper:
            self.conv5 = nn.Conv2d(ch,ch,3,padding='same')  # 9x9xch
            self.bn5 = nn.BatchNorm2d(ch)  

            self.conv6 = nn.Conv2d(ch,ch,3,padding='same') # 7x7xch
            self.bn6 = nn.BatchNorm2d(ch)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9 * 9 * ch, 50)
        self.bn_fc = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,7)

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        # STN & localization network
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
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(self.pool2(out))

        # if using deformable convolution (only for 3/4 layer)
        if self.de_conv:
            out = F.relu(self.de_conv3(out))
            out = self.bn3(out)
            out = self.bn4(self.de_conv4(out))
            out = F.relu(self.pool4(out))
        else:
            out = F.relu(self.conv3(out))
            out = self.bn3(out)
            out = self.bn4(self.conv4(out))
            out = F.relu(self.pool4(out))

        # deeper
        if self.deeper:
            out = F.relu(self.conv5(out))
            out = self.bn5(out)

            out = F.relu(self.conv6(out))
            out = self.bn6(out)

        out = F.dropout(out)
        # out = out.view(-1, 1327104)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = self.bn_fc(out)
        out = self.fc2(out)

        return out

