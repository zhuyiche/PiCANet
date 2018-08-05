import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
from module import Renet, AttentionGlobal, AttentionLocal

class Encoder(nn.Module):
    def __init__(self, inchannel=None, outchannel=None):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, 64, 7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pooling1 = nn.MaxPool2d(stride=2, kernel_size=3)
        self.inchannel = inchannel

        class _BasicShortcut(nn.Module):
            def __init__(self, inchannel, outchannel, stride=None):
                super(_BasicShortcut, self).__init__()
                if stride == False:
                    self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=1,
                                          stride=1, padding=0, bias=False)
                else:
                    self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=1,
                                          stride=2, padding=0, bias=False)
                self.bn = nn.BatchNorm2d(outchannel)

            def forward(self, *input):
                x = input[0]
                x = self.conv(x)
                out = self.bn(x)
                return out

        class _ElongShortcut(nn.Module):
            def __init__(self, inchannel, medchannel, outchannel, stage, stride=None, dilation=None):
                super(_ElongShortcut, self).__init__()
                if stage == 2 or stage == 3:
                    if stride == False:
                        self.conv1 = nn.Conv2d(inchannel, medchannel, kernel_size=1,
                                               padding=0, stride=1, bias=False)
                    else:
                        self.conv1 = nn.Conv2d(inchannel, medchannel, kernel_size=2,
                                               padding=0, stride=1, bias=False)
                    self.shortcut = nn.Sequential(
                        nn.BatchNorm2d(medchannel),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(medchannel, medchannel, kernel_size=3, padding=1, stride=1, bias=False),
                        nn.BatchNorm2d(medchannel),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(medchannel, outchannel, kernel_size=1, padding=0, stride=1),
                        nn.BatchNorm2d(outchannel)
                    )
                elif stage == 4:
                    self.conv1 = nn.Conv2d(inchannel, medchannel, kernel_size=1,
                                           padding=0, stride=1, bias=False)
                    self.shortcut = nn.Sequential(
                        nn.BatchNorm2d(medchannel),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(medchannel, medchannel, kernel_size=3, padding=2,
                                  stride=1, dilation=dilation, bias=False),
                        nn.BatchNorm2d(medchannel),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(medchannel, outchannel, kernel_size=1,
                                  padding=0, stride=1, bias=False),
                        nn.BatchNorm2d(outchannel)
                    )
                elif stage == 5:
                    self.conv1 = nn.Conv2d(inchannel, medchannel, kernel_size=1,
                                           padding=0, stride=1, bias=False)
                    self.shortcut = nn.Sequential(
                        nn.BatchNorm2d(medchannel),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(medchannel, medchannel, kernel_size=3, padding=4,
                                  stride=1, dilation=dilation, bias=False),
                        nn.BatchNorm2d(medchannel),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(medchannel, outchannel, kernel_size=1,
                                  padding=0, stride=1, bias=False),
                        nn.BatchNorm2d(outchannel)
                    )

            def forward(self, *input):
                x = input[0]
                x = self.conv1(x)
                out = self.shortcut(x)
                return out
        # Default inchannel is 64, outchannel is 2048.
        # Stage 2
        self.shortcut_2a_1 = _BasicShortcut(inchannel, inchannel*4, stride=False)
        self.shortcut_2a_2 = _ElongShortcut(inchannel, inchannel, inchannel*4, stage=2)
        self.shortcut_2bc = _ElongShortcut(inchannel*4, inchannel, inchannel*4, stage=2)
        # Stage 3
        self.shortcut_3a_1 = _BasicShortcut(inchannel*4, inchannel*8, stride=True)
        self.shortcut_3a_2 = _ElongShortcut(inchannel*4, inchannel*2, inchannel*8,
                                            stride=True, stage=3)
        self.shortcut_3bcd = _ElongShortcut(inchannel*8, inchannel*2, inchannel*8, stage=3)
        # Stage 4
        self.shortcut_4a_1 = _BasicShortcut(inchannel*8, inchannel*16, stride=False)
        self.shortcut_4a_2 = _ElongShortcut(inchannel*8, inchannel*4, inchannel*16,
                                            stride=False, stage=4)
        self.shortcut_4bcdef = _ElongShortcut(inchannel*16, inchannel*4, inchannel*16,
                                              stride=False, dilation=2, stage=4)
        # Stage 5
        self.shortcut_5a_1 = _BasicShortcut(inchannel*16, inchannel*32, stride=False)
        self.shortcut_5a_2 = _ElongShortcut(inchannel*16, inchannel*8, inchannel*32,
                                            stride=False, dilation=4, stage=5)
        self.shortcut_5bc = _ElongShortcut(inchannel*32, inchannel*8, inchannel*32,
                                           stride=False, dilation=4, stage=5)

    def forward(self, *input):
        x = input[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling1(x)
        # Stage 2
        x_branch2a_1 = self.shortcut_2a_1(x)
        x_branch2a_2 = self.shortcut_2a_2(x)
        x_branch2a = torch.cat((x_branch2a_1, x_branch2a_2), dim=0)
        x_branch2a = self.relu(x_branch2a)

        x_branch2b = self.shortcut_2bc(x)
        x_branch2b = torch.cat((x_branch2b, x_branch2a), dim=0)
        x_branch2b = self.relu(x_branch2b)

        x_branch2c = self.shortcut_2bc(x)
        x_branch2c = torch.cat((x_branch2c, x_branch2b), dim=0)
        x_branch2c = self.relu(x_branch2c)
        # Stage 3
        x_branch3a_1 = self.shortcut_3a_1(x_branch2c)
        x_branch3a_2 = self.shortcut_3a_2(x_branch2c)
        x_branch3a = torch.cat((x_branch3a_1, x_branch3a_2), dim=0)

        x_branch3b = self.shortcut_3bcd(x_branch3a)
        x_branch3b = torch.cat((x_branch3a, x_branch3b), dim=0)
        x_branch3b = self.relu(x_branch3b)

        x_branch3c = self.shortcut_3bcd(x_branch3b)
        x_branch3c = torch.cat((x_branch3c, x_branch3b), dim=0)
        x_branch3c = self.relu(x_branch3c)

        x_branch3d = self.shortcut_3bcd(x_branch3c)
        x_branch3d = torch.cat((x_branch3d, x_branch3c), dim=0)
        x_branch3d = self.relu(x_branch3d)
        # Stage 4
        x_branch4a_1 = self.shortcut_4a_1(x_branch3d)
        x_branch4a_2 = self.shortcut_4a_2(x_branch3d)
        x_branch4a = torch.cat((x_branch4a_1, x_branch4a_2), dim=0)
        x_branch4a = self.relu(x_branch4a)

        x_branch4b = self.shortcut_4bcdef(x_branch4a)
        x_branch4b = torch.cat((x_branch4b, x_branch4a), dim=0)
        x_branch4b = self.relu(x_branch4b)

        x_branch4c = self.shortcut_4bcdef(x_branch4b)
        x_branch4c = torch.cat((x_branch4c, x_branch4b), dim=0)
        x_branch4c = self.relu(x_branch4c)

        x_branch4d = self.shortcut_4bcdef(x_branch4c)
        x_branch4d = torch.cat((x_branch4d, x_branch4c), dim=0)
        x_branch4d = self.relu(x_branch4d)

        x_branch4e = self.shortcut_4bcdef(x_branch4d)
        x_branch4e = torch.cat((x_branch4e, x_branch4d), dim=0)
        x_branch4e = self.relu(x_branch4e)
        # Stage 5
        x_branch5a_1 = self.shortcut_5a_1(x_branch4e)
        x_branch5a_2 = self.shortcut_5a_2(x_branch4e)
        x_branch5a = torch.cat((x_branch5a_1, x_branch5a_2), dim=0)
        x_branch5a = self.relu(x_branch5a)

        x_branch5b = self.shortcut_5bc(x_branch5a)
        x_branch5b = torch.cat((x_branch5b, x_branch5a), dim=0)
        x_branch5b = self.relu(x_branch5b)

        x_branch5c = self.shortcut_5bc(x_branch5b)
        x_branch5c = torch.cat((x_branch5b, x_branch5c), dim=0)
        x_branch5c = self.relu(x_branch5c)

        return x_branch2c, x_branch3d, x_branch4e, x_branch5c


class Decoder(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, 64, 7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pooling1 = nn.MaxPool2d(stride=2, kernel_size=3)

        self.encode = Encoder(inchannel=64)
        self.attention5 = AttentionGlobal(patch_size=1, inplane=2048,
                                         renet_outplane=100, renetconv_channel=256)
        self.attention5 = AttentionGlobal(patch_size=1, inplane=1024,
                                          renet_outplane=100, renetconv_channel=256)
        self.conv5 = nn.Conv2d(inchannel, 1024, 1)
        self.conv4_b4 = nn.Conv2d(1024, 2014, 1)
        self.conv4_1 = nn.Conv2d(1024, 512, 1)
        self.conv3_b4 = nn.Conv2d(512, 512, 1)
        self.pool3_att_1 = nn.Conv2d(512, 128, kernel_size=7, padding=6, dilation=2)
        self.pool3_att_2 = nn.Conv2d(128, 49, 1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(49)

    def forward(self, *input):
        x = input[0]
        encode_branch2, encode_branch3, encode_branch4, encode_branch5 = self.encode(x)
        att5 = self.attention5(encode_branch5)
        decode_branch5 = torch.cat((encode_branch5, att5), dim=0)
        decode_branch5 = self.conv5(decode_branch5)
        decode_branch5 = self.bn5(decode_branch5)
        decode_branch5 = self.relu(decode_branch5)

        # Concat then do one conv
        decode_branch4 = torch.cat((encode_branch4, decode_branch5), dim=0)
        decode_branch4 = self.conv4(decode_branch4)
        decode_branch4 = self.relu(decode_branch4)

        # Global attention for branch4
        att4 = self.attention4(decode_branch4)
        decode_branch4 = torch.cat((decode_branch4, att4), dim=0)
        decode_branch4 = self.conv4_1(decode_branch4)
        decode_branch4 = self.bn4(decode_branch4)
        decode_branch4 = self.relu(decode_branch4)

        # Stage 3
        decode_branch3 = torch.cat((decode_branch4, encode_branch3), dim=0)
        decode_branch3 = self.conv3(decode_branch3)
        decode_branch3 = self.relu(decode_branch3)
        decode_branch3 = self.pool3_att_1(decode_branch3)
        decode_branch3 = self.relu(decode_branch3)
        decode_branch3 = self.pool3_att_2(decode_branch3)
        decode_branch3 = self.bn3(decode_branch3)

        # Local attention for branch3
