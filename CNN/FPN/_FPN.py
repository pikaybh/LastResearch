import torchvision
from torchvision.datasets import VOCDetection
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from PIL import Image, ImageDraw, ImageFont
import os
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Optional, Tuple, List
import warnings
import tarfile
import collections
import numpy as np
import math
import matplotlib.pyplot as plt
import imageio

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from torch import optim
import albumentations as A
# from albumentations.pytorch import ToTensor
import os
import time

original = imageio.imread('../data/test.jpg')

# from [H, W, C] to [C, H, W]
# transposed_image = original.transpose((2, 0, 1))
# add batch dim
# transposed_image = np.expand_dims(transposed_image, 2)

rgb_image = torchvision.io.read_image('../data/test.jpg', torchvision.io.ImageReadMode.UNCHANGED)
# grayscaled_image = torchvision.io.read_image('../data/test.jpg', torchvision.io.ImageReadMode.GRAY)
# print(grayscaled_image)
transposed_image = np.expand_dims(rgb_image, 3)
transposed_image = transposed_image.transpose((3, 0, 1, 2))

image_d = torch.FloatTensor(transposed_image)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# BottleNeck of ResNet
class Bottleneck(nn.Module):
    expand = 4

    def __init__(self, in_channels, inner_channels, NL, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.conv2 = nn.Conv2d(inner_channels, inner_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inner_channels)
        self.conv3 = nn.Conv2d(inner_channels, inner_channels*self.expand, 1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(inner_channels*self.expand)
        self.relu = nn.ReLU()

        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != inner_channels*self.expand:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels*self.expand, 1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_channels*self.expand)
            )

        self.relu = nn.ReLU()

        self.NL = NL

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.bn3(self.conv3(x2))
        x4 = self.relu(x3 + self.downsample(x))
        output = x4
        # print("[x] ", x.shape, "\n[x1] ", x1.shape, "\n[x2] ", x2.shape, "\n[x3] ", x3.shape, "\n[x4] ", x4.shape)

        # x = x.permute(2, 3, 0, 1)
        # x1 = x1.permute(2, 3, 0, 1)
        # x2 = x2.permute(2, 3, 0, 1)
        # x3 = x3.permute(2, 3, 0, 1)
        # x4 = x4.permute(2, 3, 0, 1)
        # print("[x1] ", x1.shape)
        
        x = x.permute(0, 1, 2, 3)[0]
        x1 = x1.permute(0, 1, 2, 3)[0]
        x2 = x2.permute(0, 1, 2, 3)[0]
        x3 = x3.permute(0, 1, 2, 3)[0]
        x4 = x4.permute(0, 1, 2, 3)[0]
        print("[x1] ", x1)
        
        x = x.permute(0, 1, 2)[0]
        x1 = x1.permute(0, 1, 2)[0]
        x2 = x2.permute(0, 1, 2)[0]
        x3 = x3.permute(0, 1, 2)[0]
        x4 = x4.permute(0, 1, 2)[0]

        x = x.detach().numpy()
        x1 = x1.detach().numpy()
        x2 = x2.detach().numpy()
        x3 = x3.detach().numpy()
        x4 = x4.detach().numpy()

        number_of_pic = (3, 2)
        plt.figure('ResNet layer c{}'.format(self.NL+1), figsize=(10, 10), )
        plt.subplot(number_of_pic[0], number_of_pic[1], 1)
        plt.title("test image")
        plt.imshow(original)
        plt.subplot(number_of_pic[0], number_of_pic[1], 2)
        plt.title("x")
        plt.imshow(x)
        plt.subplot(number_of_pic[0], number_of_pic[1], 3)
        plt.title("relu")
        plt.imshow(x1)
        plt.subplot(number_of_pic[0], number_of_pic[1], 4)
        plt.title("relu")
        plt.imshow(x2)
        plt.subplot(number_of_pic[0], number_of_pic[1], 5)
        plt.title("bn3")
        plt.imshow(x3)
        plt.subplot(number_of_pic[0], number_of_pic[1], 6)
        plt.title("relu")
        plt.imshow(x4)
        # plt.show()
        # plt.savefig('./graphs/block-{}.png'.format(), dpi=300)

        return output

# check
def test():
    x = torch.randn(3, 1920,13,13).to(device)
    net = Bottleneck(x.size(1), x.size(1)).to(device)
    # output = net(x)
    output = net(image_d)
    print(output.size())


# FPN은 ResNet의 피쳐맵에서 multi-scale로 특징을 추출합니다.
class FPN(nn.Module):
    def __init__(self, num_blocks):
        super(FPN, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False) # 300x300
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1) # 150x150

        # Bottom-up layers and ResNet
        # PyTorch 공식 홈페이지 ResNet 구현 코드와 변수명이 동일해야, pre-trained model을 불러와서 사용할 수 있습니다.
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)  # c2, 150x150
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)  # c3 75x75
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2) # c4 38x38
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2) # c5
        self.conv6 = nn.Conv2d(2048, 256, 3, stride=2, padding=1)    # p6
        self.conv7 = nn.Sequential(                                  # p7
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1)
        )

        # Lateral layers
        self.lateral_1 = nn.Conv2d(2048, 256, 1, stride=1, padding=0)
        self.lateral_2 = nn.Conv2d(1024, 256, 1, stride=1, padding=0)
        self.lateral_3 = nn.Conv2d(512, 256, 1, stride=1, padding=0)

        # Top-down layers
        self.top_down_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.top_down_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_2 = nn.Upsample(size=(75, 75), mode='bilinear', align_corners=False)

        # try:
        #     self.upsample_2 = nn.Upsample(size=(75,75), mode='bilinear', align_corners=False) # size=(75,75)를 지정해야 합니다.
        # except Exception() as e:
        #     print("[ERROR] ", e)

        #     try:
        #          self.upsample_2 = nn.Upsample(size=(180,180), mode='bilinear', align_corners=False)
        #     except Exception() as e:
        #         print("[ERROR] ", e)

        #         try:
        #             self.upsample_2 = nn.Upsample(size=(240,240), mode='bilinear', align_corners=False)
        #         except Exception() as e:
        #             print("[ERROR] ", e)

    def forward(self, x):
        # Feature extractor(ResNet)
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # FPN
        p6 = self.conv6(c5)
        p7 = self.conv7(p6)
        p5 = self.lateral_1(c5)
        p4 = self.top_down_1(self.upsample_1(p5) + self.lateral_2(c4))
        # print("\n[x] ", x.shape, 
        # "\n\n[c1] ", c1.shape, 
        # "\n[c2] ", c2.shape, 
        # "\n[c3] ", c3.shape, 
        # "\n[c4] ", c4.shape, 
        # "\n[c5] ", c5.shape,  
        # "\n[p4] ", p4.shape, 
        # "\n[p5] ", p5.shape, 
        # "\n[p6] ", p6.shape, 
        # "\n[p7] ", p7.shape)
        print("\n[x] ", x.size(), 
        "\n\n[c1] ", c1.size(), 
        "\n[c2] ", c2.size(), 
        "\n[c3] ", c3.size(), 
        "\n[c4] ", c4.size(), 
        "\n[c5] ", c5.size(),  
        "\n[p4] ", p4.size(), 
        "\n[p5] ", p5.size(), 
        "\n[p6] ", p6.size(), 
        "\n[p7] ", p7.size())
        p3 = self.top_down_2(self.upsample_2(p4) + self.lateral_3(c3))  # !!!!!!!!!!!!!!

        x = x.permute(0, 1, 2, 3)[0, 1]
        c1 = c1.permute(0, 1, 2, 3)[0, 1]
        c2 = c2.permute(0, 1, 2, 3)[0, 1]
        c3 = c3.permute(0, 1, 2, 3)[0, 1]
        c4 = c4.permute(0, 1, 2, 3)[0, 1]
        c5 = c5.permute(0, 1, 2, 3)[0, 1]
        p3 = p3.permute(0, 1, 2, 3)[0, 1]
        p4 = p4.permute(0, 1, 2, 3)[0, 1]
        p5 = p5.permute(0, 1, 2, 3)[0, 1]
        p6 = p6.permute(0, 1, 2, 3)[0, 1]
        p7 = p7.permute(0, 1, 2, 3)[0, 1]

        x = x.detach().numpy()
        c1 = c1.detach().numpy()
        c2 = c2.detach().numpy()
        c3 = c3.detach().numpy()
        c4 = c4.detach().numpy()
        c5 = c5.detach().numpy()
        p3 = p3.detach().numpy()
        p4 = p4.detach().numpy()
        p5 = p5.detach().numpy()
        p6 = p6.detach().numpy()
        p7 = p7.detach().numpy()

        number_of_pic = (3, 5)
        plt.figure('FPN', figsize=(10, 10))
        plt.subplot(number_of_pic[0], number_of_pic[1], 1)
        plt.title("test image")
        plt.imshow(original)
        plt.subplot(number_of_pic[0], number_of_pic[1], 2)
        plt.title("x")
        plt.imshow(x)
        plt.subplot(number_of_pic[0], number_of_pic[1], 6)
        plt.title("c1")
        plt.imshow(c1)
        plt.subplot(number_of_pic[0], number_of_pic[1], 7)
        plt.title("c2")
        plt.imshow(c2)
        plt.subplot(number_of_pic[0], number_of_pic[1], 8)
        plt.title("c3")
        plt.imshow(c3)
        plt.subplot(number_of_pic[0], number_of_pic[1], 9)
        plt.title("c4")
        plt.imshow(c4)
        plt.subplot(number_of_pic[0], number_of_pic[1], 10)
        plt.title("c5")
        plt.imshow(c5)
        plt.subplot(number_of_pic[0], number_of_pic[1], 11)
        plt.title("p3")
        plt.imshow(p3)
        plt.subplot(number_of_pic[0], number_of_pic[1], 12)
        plt.title("p4")
        plt.imshow(p4)
        plt.subplot(number_of_pic[0], number_of_pic[1], 13)
        plt.title("p5")
        plt.imshow(p5)
        plt.subplot(number_of_pic[0], number_of_pic[1], 14)
        plt.title("p6")
        plt.imshow(p6)
        plt.subplot(number_of_pic[0], number_of_pic[1], 15)
        plt.title("p7")
        plt.imshow(p7)
        plt.show()
        # plt.savefig('./graphs/block-{}.png'.format(str(self.a)), dpi=300)

        return p3, p4, p5, p6, p7

    def _make_layer(self, inner_channels, num_block, stride):
        strides = [stride] + [1] * (num_block-1)
        layers = []
        # for stride in strides:
        for NL, stride in enumerate(strides):
            layers.append(Bottleneck(self.in_channels, inner_channels, stride=stride, NL=NL))
            self.in_channels = inner_channels*Bottleneck.expand
        return nn.Sequential(*layers)

def FPN50(): # ResNet-50
    return FPN([3,4,6,3])


# FPN 출력값을 입력으로 받아 예측을 수행합니다.
class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=20):
        super().__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors*4) # 바운딩 박스 좌표 예측
        self.cls_head = self._make_head(self.num_anchors*self.num_classes) # 바운딩 박스 클래스 예측

    def forward(self, x):
        # p3: batch, channels, H, W
        fms = self.fpn(x) # p3, p4, p5, p6, p7
        loc_preds = []
        cls_preds = []
        for fm in fms: # fpn 출력값에 classifier 추가
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes) # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)


            ###########################################################
            x = x.permute(0, 1, 2, 3)[0, 1]
            x = x.detach().numpy()
            number_of_pic = (3, 5)
            plt.figure('FPN', figsize=(10, 10))
            plt.subplot(number_of_pic[0], number_of_pic[1], 1)
            plt.title("test image")
            plt.imshow(original)
            ##########################################################

        return torch.cat(loc_preds,1), torch.cat(cls_preds,1)

    def _make_head(self, out_channels): # 예측을 수행하는 Layer 생성
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256,256,3, stride=1, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(256, out_channels, 3, stride=1, padding=1)) # (batch,9*4,H,W) or (batch,9*20,H,W) 
        return nn.Sequential(*layers)

    def freeze_bn(self): # pre-trained model을 사용하므로, BN freeze
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

# check
if __name__ == '__main__':
    # --- ResNet --- #
    # test()

    # --- FPN --- #
    # x = torch.randn(3, 3, 600, 600).to(device)
    # x = torch.randn(3, 3, 1920, 1920).to(device)  #
    # model = FPN50().to(device)  #
    # outputs = model(x)
    # outputs = model(image_d)  #
    # for output in outputs:  #
    #     print(output.size())  #

    # --- RetinaNet --- #
    x = torch.randn(10,3,600,600).to(device)
    model = RetinaNet().to(device)
    loc_preds, cls_preds = model(x)
    # loc_preds, cls_preds = model(image_d)
    print(loc_preds.size()) # (batch, 5 * H*W * 9, 4)
    print(cls_preds.size()) # (batch, 5 * H*W * 9, 20)

'''
[citation]
https://deep-learning-study.tistory.com/616
'''