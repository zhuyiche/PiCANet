import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time


class Encoder(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, 64, 7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pooling1 = nn.MaxPool2d(stride=2, kernel_size=3)
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
        # Stage 2
        self.shortcut_2a_1 = _BasicShortcut(64, 256, stride=False)
        self.shortcut_2a_2 = _ElongShortcut(64, 64, 256, stage=2)
        self.shortcut_2bc = _ElongShortcut(256, 64, 256, stage=2)
        # Stage 3
        self.shortcut_3a_1 = _BasicShortcut(256, 512, stride=True)
        self.shortcut_3a_2 = _ElongShortcut(256, 128, 512, stride=True, stage=3)
        self.shortcut_3bcd = _ElongShortcut(512, 128, 512, stage=3)
        # Stage 4
        self.shortcut_4a_1 = _BasicShortcut(512, 1024, stride=False)
        self.shortcut_4a_2 = _ElongShortcut(512, 256, 1024, stride=False, stage=4)
        self.shortcut_4bcdef = _ElongShortcut(1024, 256, 1024, stride=False, dilation=2, stage=4)
        # Stage 5
        self.shortcut_5a_1 = _BasicShortcut(1024, 2048, stride=False)
        self.shortcut_5a_2 = _ElongShortcut(1024, 512, 2048, stride=False, dilation=4, stage=5)
        self.shortcut_5bc = _ElongShortcut(2048, 512, 2048, stride=False, dilation=4, stage=5)

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





