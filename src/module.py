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
    def __init__(self, inchannel, LSTM_channel = 32, outchannel = 256):
        """
        patch size = 1 and LSTM channel = 256 is default setting according to the origin code.
        :param inplane: input channel size.
        :param outchannel: output channel size
        :param patch_size: num of patch to be cut.
        :param LSTM_channel: filters for LSTM.
        """

        #################
        # Warning! The outchannel should be equal to Width * Height of current feature map,
        # Please change this number manually.
        #################
        super(Renet, self).__init__()
        self.horizontal_LSTM = nn.LSTM(input_size=inchannel,
                                       hidden_size=LSTM_channel,
                                       batch_first=True,
                                       bidirectional=True)
        self.vertical_LSTM = nn.LSTM(input_size=inchannel,
                                     hidden_size=LSTM_channel,
                                     batch_first=True,
                                     bidirectional=True)
        self.conv = nn.Conv2d(LSTM_channel, outchannel, 1)
        self.bn = nn.BatchNorm2d(outchannel)

    def forward(self, *input):
        x = input[0]
        vertical_concat = []
        size = x.size()
        width, height = size[2], size[3]
        assert width == height
        x = torch.transpose(x, 1, 3)
        for i in range(width):
            h, _ = self.vertical(x[:, :, i, :])
            vertical_concat.append(h)
        x = torch.stack(vertical_concat, dim=2)
        horizontal_concat = []
        for i in range(width):
            h, _ = self.horizontal(x[:, i, :, :])
            horizontal_concat.append(h)
        x = torch.stack(horizontal_concat, dim=3)
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        out = self.bn(x)
        return out


class AttentionGlobal(nn.Module):
    """
    Global Attention module.
    """
    def __init__(self,
                 inchannel,
                 att_dilation = 3,
                 renet_LSTM_channel=256,
                 global_feature = 10):
        super(AttentionGlobal, self).__init__()
        # outplane should be height * width
        self.inchannel = inchannel
        self.global_feature = global_feature
        self.att_dilation = att_dilation
        self.renet = Renet(inchannel=inchannel, LSTM_channel=renet_LSTM_channel,
                           outchannel=global_feature**2) # Set the LSTM channel and output channel.

    def forward(self, *input):
        x = input[0]
        #print(x.size())
        size = x.size()
        assert self.inchannel == size[1]
        kernel = self.renet(x)
        kernel = F.softmax(kernel, 1) #do softmax along channel axis.
        kernel = kernel.reshape(size[0] * size[2] * size[3], 1, 1, self.global_feature, self.global_feature)
        x = torch.unsqueeze(x, 0)
        x = F.conv3d(input=x, weight=kernel, bias=None, stride=1,
                     padding=0, dilation=(1, self.att_dilation, self.att_dilation), groups=size[0])
        out = torch.reshape(x, (size[0], self.inchannel, size[2], size[3]))
        return out


class _AttentionLocal_Conv(nn.Module):
    def __init__(self, inchannel, conv1_filter, conv1_dilation, conv1_outchannel,
                 conv_last_outchannel=49,
                 num_conv=2):
        """

        :param inchannel: input channel.
        :param conv1_filter: conv1 filter, current is 7
        :param conv1_dilation: conv1 dilation, current is 2
        :param conv1_outchannel: conv1 output channel, current is 128
        :param conv2_filter:  conv2 output filter, current is 1
        :param conv2_dilation: conv2 dilation, current 1.
        :param conv2_outchannel: conv2 output channel, current 49
        :param num_conv: Number of convolution layer before softmax, current number according to paper is 2.
        """
        super(_AttentionLocal_Conv, self).__init__()
        self.num_conv = num_conv
        self.padding1 = (conv1_dilation * (conv1_filter-1) - 1) /2
        self.padding2 = (1 * (1 - 1) - 1) / 2
        self.conv1 = nn.Conv2d(inchannel, conv1_outchannel, conv1_filter,
                               dilation=conv1_dilation, padding=self.padding1)
        self.bn1 = nn.BatchNorm2d(conv1_outchannel)
        self.conv_last = nn.Conv2d(conv1_outchannel, conv_last_outchannel, 1,
                               dilation=1, padding=0)
        self.bn_last = nn.BatchNorm2d(conv_last_outchannel)

    def forward(self, *input):
        x = input[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        return x


class AttentionLocal(nn.Module):
    """
    Local Attention module.
    """
    def __init__(self, inchannel, local_feature):
        """

        :param starting_point: left upper corner point
        :param width: Width of wanted feature map
        :param height: Height of wanted feature map
        :param kernels:
        :param dilation:
        """
        super(AttentionLocal, self).__init__()
        self.local_feature = local_feature
        self.inchannel = inchannel
        self._conv = _AttentionLocal_Conv(inchannel, conv1_filter=7, conv1_dilation=2,
                                          conv1_outchannel=256,
                                          conv_last_outchannel=local_feature**2)

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self._conv(x)
        kernel = F.softmax(kernel, dim=1)
        kernel = kernel.reshape(size[0] * size[2] * size[3], 1, 1, self.local_feature, self.local_feature)
        x = torch.unsqueeze(x, 0)
        x = F.conv3d(input=x, weight=kernel, bias=None, stride=1,
                     padding=0, dilation=(1, self.att_dilation, self.att_dilation), groups=size[0])
        out = torch.reshape(x, (size[0], self.inchannel, size[2], size[3]))
        return out