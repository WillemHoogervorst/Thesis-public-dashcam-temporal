import torch
from torch import nn

# local imports
from models.layers_4D import *


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, ch, dp, wh, printing=False, cfg='models/arguments.yaml'):
        super(NeuralNetwork, self).__init__()
        # self.yaml_file = Path(cfg).name
        # with open(cfg, encoding='ascii', errors='ignore') as f:
        #         self.yaml = yaml.safe_load(f)  # model dict
        self.ch = ch # input channels
        self.dp = dp # input depth (window length / stride)
        self.wh = wh # input width height
        self.show_tensor_shapes = printing
        wh_out = int(20 / (640 / self.wh)) # output width height
        if wh_out < 1:
            print("Input image too small!")

        # Define model
        # backbone
        k1=5 if self.dp>7 else 3
        k2=5 if self.dp>9 else 3
        k3=3 if self.dp>5 else 1
        self.backbone1 = nn.Sequential(
            Conv(3, self.ch, (k1,6,6), (1,2,2), (0,2,2)),
            Conv(self.ch, self.ch*2, (k2,3,3), (1,2,2), (0,1,1)),
            C3(self.ch*2, self.ch*2, 1, d=1),
            Conv(self.ch*2, self.ch*4, (k3,3,3), (1,2,2), (0,1,1)),
            C3(self.ch*4, self.ch*4, 2, d=1)
        )
        self.backbone2 = nn.Sequential(
            Conv(self.ch*4, self.ch*8, (1,3,3), (1,2,2), (0,1,1)),
            C3(self.ch*8, self.ch*8, 3)
        )
        self.backbone3 = nn.Sequential(
            Conv(self.ch*8, self.ch*16, (1,3,3), (1,2,2), (0,1,1)),
            C3(self.ch*16, self.ch*16, 1),
            SPPF(self.ch*16, self.ch*16, 5)
        )
        # neck
        self.conv1 = Conv(self.ch*16, self.ch*8, 1, 1)
        self.upsample1 = nn.Upsample(None, (1, 2, 2), 'nearest')
        self.c3_1 = C3(self.ch*16, self.ch*8, 1, False)
        self.conv2 = Conv(self.ch*8, self.ch*4, 1, 1)
        self.upsample2 = nn.Upsample(None, (1, 2, 2), 'nearest')
        self.c3_2 = C3(self.ch*8, self.ch*4, 1, False)
        self.conv3 = Conv(self.ch*4, self.ch*4, (1,3,3), (1,2,2), (0,1,1))
        self.c3_3 = C3(self.ch*8, self.ch*8, 1, False)
        self.conv4 = Conv(self.ch*8, self.ch*8, (1,3,3), (1,2,2), (0,1,1))
        self.c3_4 = C3(self.ch*16, self.ch*16, 1, False)
        # head
        # self.detect = Detect()
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
        if self.show_tensor_shapes: print('input: ', x.shape)
        bb1 = self.backbone1(x)
        if self.show_tensor_shapes: print('bb1: ', bb1.shape)
        bb2 = self.backbone2(bb1)
        if self.show_tensor_shapes: print('bb2: ', bb2.shape)
        bb3 = self.backbone3(bb2)
        if self.show_tensor_shapes: print('bb3: ', bb3.shape)
        h1 = self.conv1(bb3)
        if self.show_tensor_shapes: print('h1: ', h1.shape)
        h2 = self.upsample1(h1)
        if self.show_tensor_shapes: print(h2.shape, bb2.shape)
        concat1 = torch.cat((h2, bb2), dim=1) # ch=256+256
        h3 = self.c3_1(concat1)
        if self.show_tensor_shapes: print('h3: ', h3.shape)
        h4 = self.conv2(h3)
        if self.show_tensor_shapes: print('h4: ', h4.shape)
        h5 = self.upsample2(h4)
        if self.show_tensor_shapes: print(h5.shape, bb1.shape)
        concat2 = torch.cat((h5, bb1), dim=1) # ch=128+128
        h6 = self.c3_2(concat2)
        if self.show_tensor_shapes: print('h6: ', h6.shape)
        h7 = self.conv3(h6)
        if self.show_tensor_shapes: print(h7.shape, h4.shape)
        concat3 = torch.cat((h7, h4), dim=1) # ch=128+128
        h8 = self.c3_3(concat3)
        if self.show_tensor_shapes: print('h8: ', h8.shape)
        h9 = self.conv4(h8)
        if self.show_tensor_shapes: print(h9.shape, h1.shape)
        concat4 = torch.cat((h9, h1), dim=1) # ch=256+256
        h10 = self.c3_4(concat4)
        if self.show_tensor_shapes: print('h10: ', h10.shape)
        detect = self.head(h10)
        if self.show_tensor_shapes: print('head: ', detect.shape)
        self.show_tensor_shapes = False

        return detect
