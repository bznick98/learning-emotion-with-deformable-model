import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dcn import DeformableConv2d

class Deep_Emotion(nn.Module):
    def __init__(self, wider=False, deeper=False, de_conv=False, input_224=False, drop=0, n_drop=1):
        '''
        Deep_Emotion (wider)
        input: (NxCx48x48) (can accept 224x224 input if input_224 set to True)
            - wider: if enabled, channel=10 will be replaced by channel=64
            - deeper: if enabled, 2 more conv layers (does not give too much difference)
            - de_conv: if enabled, conv layer 3,4 will be replaced by deformable convolution (slower computation)
            - drop: dropout rate when training
        '''
        super(Deep_Emotion,self).__init__()
        self.wider = wider
        self.deeper = deeper
        self.de_conv = de_conv
        self.n_drop = n_drop
        if wider:
            ch = 64
        else:
            ch = 10

        self.conv1 = nn.Conv2d(1,ch,3)      # 46x46xch  /   222x222xch
        self.bn1 = nn.BatchNorm2d(ch)     
        self.conv2 = nn.Conv2d(ch,ch,3)     # 44x44xch  /   220x220xch
        self.bn2 = nn.BatchNorm2d(ch)     
        self.pool2 = nn.MaxPool2d(2,2)      # 22x22xch  /   110x110xch

        self.conv3 = nn.Conv2d(ch,ch,3)    # 20x20xch   /   108x108xch
        self.bn3 = nn.BatchNorm2d(ch) 
        self.conv4 = nn.Conv2d(ch,ch,3)    # 18x18xch   /   106x106xch
        self.bn4 = nn.BatchNorm2d(ch)  
        self.pool4 = nn.MaxPool2d(2,2)     # 9x9xch     /   53x53xch

        # replace conv3/4 with deformable convolution
        if de_conv:
            self.conv3 = DeformableConv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=0)
            self.conv4 = DeformableConv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=0)

        self.dropout = nn.Dropout(drop)    # use nn.Dropout instead of F.dropout

        # deeper network
        if deeper:
            self.conv5 = nn.Conv2d(ch,ch,3,padding='same')  # 9x9xch
            self.bn5 = nn.BatchNorm2d(ch)  

            self.conv6 = nn.Conv2d(ch,ch,3,padding='same') 
            self.bn6 = nn.BatchNorm2d(ch)

        self.flatten = nn.Flatten()
        if input_224:
            # handle 224x224 input
            self.fc1 = nn.Linear(53 * 53 * ch, 50)
        else:
            # handle 48x48 input
            self.fc1 = nn.Linear(9 * 9 * ch, 50)
        self.bn_fc = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,7)

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        # STN & localization network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),     # 42x42xch  /   # 218x218xch
            nn.MaxPool2d(2, stride=2),          # 21x21xch  /   # 109x109xch
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),    # 16x16xch  /   # 104x104xch
            nn.MaxPool2d(2, stride=2),          # 8x8xch    /   # 52x52xch
            nn.ReLU(True)
        )

        if input_224:
            flat_len = 52 * 52 * 10 # 224x224 input
        else:
            flat_len = 8 * 8 * 10   # 48x48 input
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_len, 32),
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
        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        out = self.pool2(out)

        out = F.relu(self.conv3(out))
        out = self.bn3(out)
        out = F.relu(self.conv4(out))
        out = self.bn4(out)
        out = self.pool4(out)

        # deeper
        if self.deeper:
            out = F.relu(self.conv5(out))
            out = self.bn5(out)

            out = F.relu(self.conv6(out))
            out = self.bn6(out)

        # out = out.view(-1, 1327104)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = self.bn_fc(out)
        out = self.dropout(out)
        out = self.fc2(out)

        if self.n_drop == 2:
            out = self.dropout(out)

        return out



class Deep_Emotion224(nn.Module):
    def __init__(self, de_conv=False, drop=0, n_drop=1):
        '''
        Deep_Emotion, designed for 224x224 input, channels are wider than baseline Deep_Emotion
        input: (NxCx224x224)
            - de_conv: if enabled, conv layer 3,4,5 will be replaced by deformable convolution (slower computation)
            - drop: dropout rate when training
        '''
        super().__init__()

        self.de_conv = de_conv
        self.n_drop = n_drop

        self.conv1 = nn.Conv2d(1,32,3)      # 222x222x32
        self.bn1 = nn.BatchNorm2d(32)     
        self.conv2 = nn.Conv2d(32,32,3)     # 220x220x32
        self.bn2 = nn.BatchNorm2d(32)     
        self.pool2 = nn.MaxPool2d(2,2)      # 110x110x32

        self.conv3 = nn.Conv2d(32,64,3)    # 108x108x64
        self.bn3 = nn.BatchNorm2d(64) 
        self.conv4 = nn.Conv2d(64,64,3)    # 106x106x64
        self.bn4 = nn.BatchNorm2d(64)  
        self.conv5 = nn.Conv2d(64,64,3)    # 104x104x64
        self.bn5 = nn.BatchNorm2d(64)  
        self.pool5 = nn.MaxPool2d(2,2)     # 52x52x64

        # replace conv3/4 with deformable convolution
        if de_conv:
            self.conv3 = DeformableConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
            self.conv4 = DeformableConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
            self.conv5 = DeformableConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.conv6 = nn.Conv2d(64,128,3)   # 50x50x128
        self.bn6 = nn.BatchNorm2d(128) 
        self.conv7 = nn.Conv2d(128,128,3)  # 48x48x128
        self.bn7 = nn.BatchNorm2d(128)  
        self.pool7 = nn.MaxPool2d(2,2)     # 24x24x128

        self.conv8 = nn.Conv2d(128,32,1)   # 24x24x32
        self.bn8 = nn.BatchNorm2d(32)  
        self.pool8 = nn.MaxPool2d(2,2)     # 12x12x32

        self.flatten = nn.Flatten()
        # handle 48x48 input
        self.fc1 = nn.Linear(12 * 12 * 32, 32)
        self.bn_fc = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(drop)
        self.fc2 = nn.Linear(32,7)

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        # STN & localization network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),     # 42x42xch  /   # 218x218xch
            nn.MaxPool2d(2, stride=2),          # 21x21xch  /   # 109x109xch
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),    # 16x16xch  /   # 104x104xch
            nn.MaxPool2d(2, stride=2),          # 8x8xch    /   # 52x52xch
            nn.ReLU(True)
        )

        flat_len = 52 * 52 * 10 # 224x224 input for stn
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_len, 32),
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

        # 1 & 2
        out = F.relu(self.conv1(out))
        out = self.bn1(out)
        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        out = self.pool2(out)
        
        # 3 & 4 & 5
        out = F.relu(self.conv3(out))
        out = self.bn3(out)
        out = F.relu(self.conv4(out))
        out = self.bn4(out)
        out = F.relu(self.conv5(out))
        out = self.bn5(out)
        out = self.pool5(out)

        # deeper 6 & 7
        out = F.relu(self.conv6(out))
        out = self.bn6(out)
        out = F.relu(self.conv7(out))
        out = self.bn7(out)
        out = self.pool7(out)

        # 8 reduce features using 1x1 conv
        out = F.relu(self.conv8(out))
        out = self.bn8(out)
        out = self.pool8(out)

        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = self.bn_fc(out)
        out = self.dropout(out)
        out = self.fc2(out)

        if self.n_drop == 2:
            out = self.dropout(out)

        return out

# ONLY FOR TESTING
if __name__ == "__main__":
    net = Deep_Emotion224(de_conv=False)
    x = torch.ones((16,1,224,224))
    out = net(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

