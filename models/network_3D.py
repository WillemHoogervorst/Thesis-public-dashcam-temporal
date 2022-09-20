import torch
from torch import nn
from models.layers_3D import *


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, ch, wh):
        super(NeuralNetwork, self).__init__()
        self.ch = ch # input channels
        self.wh = wh # input width height
        wh_out = int(20 / (640 / self.wh)) # output width height
        if wh_out < 1:
            print("Input image too small!")

        # Define model
        # backbone
        self.backbone1 = nn.Sequential(
            Conv(3, self.ch, 6, 2, 2),
            Conv(self.ch, self.ch*2, 3, 2),
            C3(self.ch*2, self.ch*2, 1),
            Conv(self.ch*2, self.ch*4, 3, 2),
            C3(self.ch*4, self.ch*4, 2)
        )
        self.backbone2 = nn.Sequential(
            Conv(self.ch*4, self.ch*8, 3, 2),
            C3(self.ch*8, self.ch*8, 3)
        )
        self.backbone3 = nn.Sequential(
            Conv(self.ch*8, self.ch*16, 3, 2),
            C3(self.ch*16, self.ch*16, 1),
            SPPF(self.ch*16, self.ch*16, 5)
        )
        # neck
        self.conv1 = Conv(self.ch*16, self.ch*8, 1, 1)
        self.upsample = nn.Upsample(None, 2, 'nearest')
        self.c3_1 = C3(self.ch*16, self.ch*8, 1, False)
        self.conv2 = Conv(self.ch*8, self.ch*4, 1, 1)
        self.c3_2 = C3(self.ch*8, self.ch*4, 1, False)
        self.conv3 = Conv(self.ch*4, self.ch*4, 3, 2)
        self.c3_3 = C3(self.ch*8, self.ch*8, 1, False)
        self.conv4 = Conv(self.ch*8, self.ch*8, 3, 2)
        self.c3_4 = C3(self.ch*16, self.ch*16, 1, False)
        # head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.ch*16*wh_out*wh_out, self.ch*8*wh_out*wh_out),
            nn.ReLU(),
            nn.Linear(self.ch*8*wh_out*wh_out, self.ch*4*wh_out*wh_out),
            nn.ReLU(),
            nn.Linear(self.ch*4*wh_out*wh_out, self.ch*2*wh_out*wh_out),
            nn.ReLU(),
            nn.Linear(self.ch*2*wh_out*wh_out, self.ch*1*wh_out*wh_out),
            nn.ReLU(),
            nn.Linear(self.ch*1*wh_out*wh_out, int(self.ch*0.5*wh_out*wh_out)),
            nn.ReLU(),
            nn.Linear(int(self.ch*0.5*wh_out*wh_out), int(self.ch*0.25*wh_out*wh_out)),
            nn.ReLU(),
            nn.Linear(int(self.ch*0.25*wh_out*wh_out), 4),
            nn.LogSoftmax(dim=0)
        )

    def forward(self, x):
        bb1 = self.backbone1(x)
        bb2 = self.backbone2(bb1)
        bb3 = self.backbone3(bb2)
        h1 = self.conv1(bb3)
        h2 = self.upsample(h1)
        concat1 = torch.cat((h2, bb2), dim=1) # ch=256+256
        h3 = self.c3_1(concat1)
        h4 = self.conv2(h3)
        h5 = self.upsample(h4)
        concat2 = torch.cat((h5, bb1), dim=1) # ch=128+128
        h6 = self.c3_2(concat2)
        h7 = self.conv3(h6)
        concat3 = torch.cat((h7, h4), dim=1) # ch=128+128
        h8 = self.c3_3(concat3)
        h9 = self.conv4(h8)
        concat4 = torch.cat((h9, h1), dim=1) # ch=256+256
        h10 = self.c3_4(concat4)
        detect = self.head(h10)
        return detect
