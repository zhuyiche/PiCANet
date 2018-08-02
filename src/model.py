import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
import time



class Renet(nn.Module):
    def __init__(self, patch_size, inplane, outplane):
        super(Renet, self).__init__()
        self.patch_size = patch_size
        self.inplane = inplane
        self.outplane = outplane
        self.horizontal_LSTM = nn.LSTM(input_size=inplane,
                                       hidden_size=inplane,
                                       batch_first=True,
                                       bidirectional=True)
        self.vertical_LSTM = nn.LSTM(input_size=2*inplane,
                                     hidden_size=inplane,
                                     batch_first=True,
                                     bidirectional=True)
        self.conv = nn.Conv2d(2*inplane, outplane, 1)

    def forward(self, *input):
        # input is (batch, channel, width, height)

        # Here we follow PiCANet which we first flip horizontally twice,
        # Then vertically twice,
        x = input[0]
        vertical_fwd_concat = []
        vertical_inv_concat = []
        horizon_fwd_concat = []
        horizon_inv_concat = []

        width, height = x[2], x[3]
        width_per_patch = width / self.patch_size
        height_per_patch = height / self.patch_size
        assert width_per_patch.is_interger()
        assert height_per_patch.is_interger()
        #######
        # use LSTM horizontally forward and backward
        for i in range(self.patch_size):
            horizon_fwd, _ = self.horizontal_LSTM(x[:, :, i: (i+1) * width_per_patch, :])
            horizon_fwd_concat.append(horizon_fwd)
        x_horizon_fwd = torch.stack(horizon_fwd_concat, dim=2)

        for i in reversed(self.patch_size):
            horizon_inv, _ = self.horizontal_LSTM(x[:, :, i: (i+1) * width_per_patch, :])
            horizon_inv_concat.append(horizon_inv)
        x_horizon_inv = torch.stack(horizon_inv_concat, dim=2)

        x = torch.concat(x_horizon_fwd, x_horizon_inv)
        #######
        # use LSTM vertically upward and downward
        for j in range(self.patch_size):
            vertical_fwd, _ = self.vertical_LSTM(x[:, :, :, j: (j+1) * height_per_patch])
            vertical_fwd_concat.append(vertical_fwd)
        x_vertical_fwd = torch.stack(vertical_fwd_concat, dim=3)

        for j in reversed(range(self.patch_size)):
            vertical_inv, _ = self.vertical_LSTM(x[:, :, :, j: (j+1) * height_per_patch])
            vertical_inv_concat.append(vertical_inv)
        x_vertical_inv = torch.stack(vertical_inv_concat, dim=3)
        x = torch.concat(x_vertical_fwd, x_vertical_inv)

        out = self.conv(x)
        return out


class Attention_Golbal(nn.Module):
    def __init__(self, patch_size, inplane, outplane, dilation=(1, 3, 3)):
        super(Attention_Golbal, self).__init__()
        # outplane should be height * width
        self.renet = Renet(patch_size, inplane, outplane)
        self.softmax = F.softmax(input, dim=1)
        self.in_channel = inplane
        self.dilation = dilation

    def forward(self, *input):
        x = input[0]
        x_size = x.size()
        x_renet = self.renet(x)
        x_renet = self.softmax(x_renet)
        x_renet = x_renet.reshape(x_size[0], )
        x_renet = x.unsqueeze(0)
        x = F.conv3d(input=x, weight=x_renet, bias=None, stride=1, padding=0,
                     dilation=self.dilation, group=x_size[0])

        return x

if __name__ == '__main__':
    from torch import IntTensor
    from torch.autograd import Variable

    a = [1, 2, 3, 4]
    for i in reversed(range(20)):
        print(i)