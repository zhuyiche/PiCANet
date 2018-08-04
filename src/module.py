import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time


def _isArrayLike(obj):
    """
    check if this is array like object.
    """
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Renet(nn.Module):
    """
    This Renet is implemented according to paper
    """
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
        self.bn = nn.BatchNorm2d(outplane)


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
        out = self.bn(out)
        return out


class AttentionGlobal(nn.Module):
    """
    Global Attention module.
    """
    def __init__(self, patch_size, inplane, outplane):
        super(AttentionGlobal, self).__init__()
        # outplane should be height * width
        self.patch_size = patch_size
        self.renet = Renet(patch_size, inplane, outplane)
        self.softmax = F.softmax(input, dim=1)
        self.in_channel = inplane

    def forward(self, *input):
        x = input[0]
        x_size = x.size()
        x_renet = self.renet(x)
        # reshape tensor of (batch, channel, w, h) to
        # (batch, 1, w, h) then do softmax
        x_renet = x_renet.view(1, 1)
        print(x_renet.size())
        x_renet = self.softmax(x_renet)
        out = x_renet + x
        return out


class AttentionLocal(nn.Module):
    """
    Local Attention module.
    """
    def __init__(self, inplane, implane, outplane, kernels, dilation=None):
        super(AttentionLocal, self).__init__()
        self.sofxmax = F.softmax(input, dim=1)
        assert _isArrayLike(kernels)
        self.pre_attention = nn.Sequential(
            nn.Conv2d(inplane, implane, kernels[0],
                      padding=kernels[0]-1, dilation=dilation[0]),
            nn.BatchNorm2d(implane),
            nn.ReLU(inplace=True),
            nn.Conv2d(implane, outplane, kernels[1],
                      padding=kernels[1]-1, dilation=dilation[1]),
            nn.BatchNorm2d(outplane)
        )

    def forward(self, *input):
        x = input[0]
        x_att = self.pre_attention(x)
        x_att = x_att.view(1,1)
        print(x_att.size())
        x_att = self.softmax(x_att)
        out = x_att + x
        return out