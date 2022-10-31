import torchvision
from torchvision.datasets import VOCDetection
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# import torchmetrics

from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Optional, Tuple, List
import warnings
import tarfile
import collections
import numpy as np
import math
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from torch import optim
import albumentations as A

import os
import time
import copy

from PIL import Image, ImageDraw
# from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout

print('Check if using Torch 1.12.0+cpu')
print('Using Torch version', torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device using', device)

emojis = {
    0: {'name': 'happy', 'file': '1F642.png'},
    1: {'name': 'laughing', 'file': '1F602.png'},
    2: {'name': 'skeptical', 'file': '1F928.png'},
    3: {'name': 'sad', 'file': '1F630.png'},
    4: {'name': 'cool', 'file': '1F60E.png'},
    5: {'name': 'whoa', 'file': '1F62F.png'},
    6: {'name': 'crying', 'file': '1F62D.png'},
    7: {'name': 'puking', 'file': '1F92E.png'},
    8: {'name': 'nervous', 'file': '1F62C.png'}
}

plt.figure(figsize=(9, 9))

for i, (j, e) in enumerate(emojis.items()):
    plt.subplot(3, 3, i + 1)
    # plt.imshow(plt.imread(os.path.join('emojis', e['file'])))
    plt.imshow(plt.imread(os.path.join('data/emojis', e['file'])))
    plt.xlabel(e['name'])
    plt.xticks([])
    plt.yticks([])
# plt.show()

## Task 3: Create Examples

for class_id, values in emojis.items():
    png_file = Image.open(os.path.join('data/emojis', values['file'])).convert('RGBA')
    png_file.load()
    new_file = Image.new("RGB", png_file.size, (255, 255, 255))
    new_file.paste(png_file, mask=png_file.split()[3])
    emojis[class_id]['image'] = new_file

class myDataGen():
    def create_example():
        lass_id = np.random.randint(0, 9)
        image = np.ones((144, 144, 3)) * 255
        row = np.random.randint(0, 72)
        col = np.random.randint(0, 72)
        image[row: row + 72, col: col + 72, :] = np.array(emojis[class_id]['image'])
        return image.astype('uint8'), class_id, (row + 10) / 144, (col + 10) / 144

    image, class_id, row, col = create_example()
    plt.figure(figsize=(9, 9)) # 
    plt.imshow(image);

    ## Task 4: Plot Bounding Boxes

    def plot_bounding_box(image, gt_coords, pred_coords=[], norm=False):
        if norm:
            image *= 255.
            image = image.astype('uint8')
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        row, col = gt_coords
        row *= 144
        col *= 144
        draw.rectangle((col, row, col + 52, row + 52), outline='green', width=3)

        if len(pred_coords) == 2:
            row, col = pred_coords
            row *= 144
            col *= 144
            draw.rectangle((col, row, col + 52, row + 52), outline='red', width=3)
        return image

    image = plot_bounding_box(image, gt_coords=[row, col])
    plt.imshow(image)
    plt.title(emojis[class_id]['name'])
    # plt.show()

    ## Task 5: Data Generator
    def data_generator(batch_size=16):
        while True:
            x_batch = np.zeros((batch_size, 144, 144, 3))
            y_batch = np.zeros((batch_size, 9))
            bbox_batch = np.zeros((batch_size, 2))

            for i in range(0, batch_size):
                image, class_id, row, col = create_example()
                x_batch[i] = image / 255.
                y_batch[i, class_id] = 1.0
                bbox_batch[i] = np.array([row, col])
            # yield {'image': x_batch}, {'class_out': y_batch, 'box_out': bbox_batch}
            return x_batch, y_batch, bbox_batch

    plt.figure(figsize=(9, 9)) #

# example, label = next(data_generator(1))
# image = example['image'][0]
# class_id = np.argmax(label['class_out'][0])
# coords = label['box_out'][0]

# image = plot_bounding_box(image, coords, norm=True)
# plt.imshow(image)
# plt.title(emojis[class_id]['name'])
# plt.show()


#########################################
train_ds = myDataGen()
val_ds = myDataGen()

# 샘플 이미지 확인
img, target, label = train_ds[2]
colors = np.random.randint(0, 255, size=(80,3), dtype='uint8') # 바운딩 박스 색상

# 시각화 함수
def show(img, targets, labels, classes=classes):
    img = to_pil_image(img)
    draw = ImageDraw.Draw(img)
    targets = np.array(targets)
    W, H = img.size

    for tg,label in zip(targets,labels):
        id_ = int(label) # class
        bbox = tg[:4]    # [x1, y1, x2, y2]

        color = [int(c) for c in colors[id_]]
        name = classes[id_]

        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=tuple(color), width=3)
        draw.text((bbox[0], bbox[1]), name, fill=(255,255,255,0))
    plt.imshow(np.array(img))

plt.figure(figsize=(10,10))
show(img, target, label)

# transforms 정의
IMAGE_SIZE = 600
scale = 1.0

# 이미지에 padding을 적용하여 종횡비를 유지시키면서 크기가 600x600 되도록 resize 합니다.
train_transforms = A.Compose([
                    A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
                    A.PadIfNeeded(min_height=int(IMAGE_SIZE*scale), min_width=int(IMAGE_SIZE*scale),border_mode=cv2.BORDER_CONSTANT),
                    ToTensor()
                    ],
                    bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                    )

val_transforms = A.Compose([
                    A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
                    A.PadIfNeeded(min_height=int(IMAGE_SIZE*scale), min_width=int(IMAGE_SIZE*scale),border_mode=cv2.BORDER_CONSTANT),
                    ToTensor()
                    ],
                    bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                    )

# transforms 적용하기
train_ds.transforms = train_transforms
val_ds.transforms = val_transforms

class DataEncoder:
    def __init__(self):
        self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.] # 피쳐맵 크기 p3 -> p7
        self.aspect_ratios = [1/2., 1/1., 2/1.]            # 앵커 박스 종횡비, w/h
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)] # 앵커 박스 scale
        self.anchor_wh = self._get_anchor_wh() # 5개의 피쳐맵 각각에 해당하는 9개의 앵커 박스 생성 

    def _get_anchor_wh(self):
        # 각 피쳐맵에서 사용할 앵커 박스 높이와 넓이를 계산합니다.
        anchor_wh = []
        for s in self.anchor_areas: # 각 피쳐맵 크기 추출
            for ar in self.aspect_ratios: # ar = w/h
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios: # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2) # [#fms, #anchors_pre_cell, 2], [5, 9, 2]

    def _get_anchor_boxes(self, input_size):
        # 피쳐맵의 모든 cell에 앵커 박스 할당
        num_fms = len(self.anchor_areas) # 5
        fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)] # 각 피쳐맵 stride 만큼 입력 크기 축소

        boxes = []
        for i in range(num_fms): # p3 ~ p7
            fm_size = fm_sizes[i] # i 번째 피쳐맵 크기 추출
            grid_size = input_size / fm_size # 입력 크기를 피쳐맵 크기로 나누어 grid size 생성
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = self._meshgrid(fm_w, fm_h) + 0.5 #[fm_h * fm_w, 2] 피쳐맵 cell index 생성
            xy = (xy*grid_size).view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, 9, 2) # anchor 박스 좌표
            wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_h, fm_w, 9, 2) # anchor 박스 높이와 너비
            box = torch.cat([xy,wh],3) # [x,y,w,h]
            boxes.append(box.view(-1,4))
        return torch.cat(boxes, 0)

    # 피쳐맵의 각 셀에 anchor 박스 생성하고, positive와 negative 할당
    def encode(self, boxes, labels, input_size):
        input_size = torch.Tensor([input_size, input_size]) if isinstance(input_size, int) else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size) # 앵커 박스 생성
        boxes = self._change_box_order(boxes, 'xyxy2xywh') # xyxy -> cxcywh

        ious = self._box_iou(anchor_boxes, boxes, order='xywh') # ground-truth와 anchor의 iou 계산
        max_ious, max_ids = ious.max(1) # 가장 높은 iou를 지닌 앵커 추출
        boxes = boxes[max_ids]

        # 앵커 박스와의 offset 계산
        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)

        # class 할당
        cls_targets = 1 + labels[max_ids]
        cls_targets[max_ious<0.5] = 0 # iou < 0.5 anchor는 negative
        ignore = (max_ious>0.4) & (max_ious<0.5) # [0.4,0.5] 는 무시
        cls_targets[ignore] = -1
        return loc_targets, cls_targets

    # encode된 값을 원래대로 복구 및 nms 진행
    def decode(self,loc_preds, cls_preds, input_size):
        cls_thresh = 0.5
        nms_thresh = 0.5

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size) # 앵커 박스 생성

        loc_xy = loc_preds[:,:2] # 결과값 offset 추출
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2] # offset + anchor
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)

        score, labels = cls_preds.sigmoid().max(1)
        ids = score > cls_thresh
        ids = ids.nonzero().squeeze()
        keep = self._box_nms(boxes[ids], score[ids], threshold=nms_thresh) # nms
        return boxes[ids][keep], labels[ids][keep]

    # cell index 생성 함수
    def _meshgrid(self, x, y, row_major=True):
        a = torch.arange(0,x)
        b = torch.arange(0,y)
        xx = a.repeat(y).view(-1,1)
        yy = b.view(-1,1).repeat(1,x).view(-1,1)
        return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)
    
    # x1,y1,x2,y2 <-> cx,cy,w,h
    def _change_box_order(self, boxes, order):
        assert order in ['xyxy2xywh','xywh2xyxy']
        boxes = np.array(boxes)
        a = boxes[:,:2]
        b = boxes[:,2:]
        a, b = torch.Tensor(a), torch.Tensor(b)
        if order == 'xyxy2xywh':
            return torch.cat([(a+b)/2,b-a+1],1) # xywh
        return torch.cat([a-b/2, a+b/2],1) # xyxy

    # 두 박스의 iou 계산
    def _box_iou(self, box1, box2, order='xyxy'):
        if order == 'xywh':
            box1 = self._change_box_order(box1, 'xywh2xyxy')
            box2 = self._change_box_order(box2, 'xywh2xyxy')
        
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(box1[:,None,:2], box2[:,:2])
        rb = torch.min(box1[:,None,2:], box2[:,2:])

        wh = (rb-lt+1).clamp(min=0)
        inter = wh[:,:,0] * wh[:,:,1]

        area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)
        area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)
        iou = inter / (area1[:,None] + area2 - inter)
        return iou

    # nms
    def _box_nms(self, bboxes, scores, threshold=0.5, mode='union'):
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]

        areas = (x2-x1+1) * (y2-y1+1)
        _, order = scores.sort(0, descending=True) # confidence 순 정렬
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.data)
                break
            i = order[0] # confidence 가장 높은 anchor 추출
            keep.append(i) # 최종 detection에 저장

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1+1).clamp(min=0)
            h = (yy2-yy1+1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr<=threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]
        return torch.LongTensor(keep)

# collate_fn
# targets에 encode를 수행하고, tensor로 변경합니다.
def collate_fn(batch):
    encoder = DataEncoder()
    imgs = [x[0] for x in batch]
    boxes = [torch.Tensor(x[1]) for x in batch]
    labels = [torch.Tensor(x[2]) for x in batch]
    h,w = 600, 600
    num_imgs = len(imgs)
    inputs = torch.zeros(num_imgs, 3, h, w)

    loc_targets = []
    cls_targets = []
    for i in range(num_imgs):
        inputs[i] = imgs[i]
        loc_target, cls_target = encoder.encode(boxes=boxes[i], labels=labels[i], input_size=(w,h))
        loc_targets.append(loc_target)
        cls_targets.append(cls_target)
    return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

# BottleNeck of ResNet
class Bottleneck(nn.Module):
    expand = 4

    def __init__(self, in_channels, inner_channels, stride=1):
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

    def forward(self, x):
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        output = self.relu(output + self.downsample(x))

        return output

# check
# def test():
#     x = torch.randn(1, 56,13,13).to(device)
#     net = Bottleneck(x.size(1), x.size(1)).to(device)
#     output = net(x)
#     print(output.size())

# test()

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
        self.upsample_2 = nn.Upsample(size=(75,75), mode='bilinear', align_corners=False) # size=(75,75)를 지정해야 합니다.

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
        p3 = self.top_down_2(self.upsample_2(p4) + self.lateral_3(c3))

        return p3, p4, p5, p6, p7

    def _make_layer(self, inner_channels, num_block, stride):
        strides = [stride] + [1] * (num_block-1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_channels, inner_channels, stride=stride))
            self.in_channels = inner_channels*Bottleneck.expand
        return nn.Sequential(*layers)

def FPN50(): # ResNet-50
    return FPN([3,4,6,3])

# check
# if __name__ == '__main__':
#     x = torch.randn(3, 3, 600, 600).to(device)
#     model = FPN50().to(device)
#     outputs = model(x)
#     for output in outputs:
#         print(output.size())

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
# if __name__ == '__main__':
#     x = torch.randn(10,3,600,600).to(device)
#     model = RetinaNet().to(device)
#     loc_preds, cls_preds = model(x)
#     print(loc_preds.size()) # (batch, 5 * H*W * 9, 4)
#     print(cls_preds.size()) # (batch, 5 * H*W * 9, 20)

# 가중치 변경
path2weight = './RetinaNet/resnet50-19c8e357.pth' # 가중치 저장할 경로
d = torch.load(path2weight) # 사전학습 가중치 읽어오기
fpn = FPN50()               # FPN50 생성
dd = fpn.state_dict()       # fpn 가중치 파일 추출
for k in d.keys():          # 사전학습 가중치로부터 가중치 추출
    if not k.startswith('fc'): # fc layer 제외
        dd[k] = d[k]        # 변수 명이 동일한 경우, 가중치 받아오기

model = RetinaNet()         # RetinaNet 가중치 초기화
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

pi = 0.01
init.constant_(model.cls_head[-1].bias, -math.log((1-pi)/pi))

model.fpn.load_state_dict(dd)  # fpn의 가중치를 사전 학습된 가중치로 변경
torch.save(model.state_dict(), 'model.pth') # 가중치 저장

# labels를 one-hot 형식으로 변경
def one_hot_embedding(labels, num_classes):
    # labels: class labels, sized [N,]
    # num_classes: 클래스 수 20
    y = torch.eye(num_classes) # [20, 20]
    np_labels = np.array(labels)
    return y[np_labels]

class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes # VOC dataset 20

    # alternative focal loss
    def focal_loss_alt(self, x, y):
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        t = t[:,1:] # 배경 제외
        t = t.cuda()

        xt = x*(2*t-1) # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        # (loc_preds, loc_targets)와 (cls_preds, cls_targets) 사이의 loss 계산
        # loc_preds: [batch_size, #anchors, 4]
        # loc_targets: [batch_size, #anchors, 4]
        # cls_preds: [batch_size, #anchors, #classes]
        # cls_targets: [batch_size, #anchors]

        # loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets)

        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()

        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        mask = pos.unsqueeze(2).expand_as(loc_preds) # [N, #anchors, 4], 객체가 존재하는 앵커박스 추출
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos, 4]
        masked_loc_targets = loc_targets[mask].view(-1, 4) # [#pos, 4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, reduction='sum')

        # cls_loss = FocalLoss(loc_preds, loc_targets)
        pos_neg = cls_targets > -1 # ground truth가 할당되지 않은 anchor 삭제
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.item(), cls_loss))
        loss = (loc_loss+cls_loss)/num_pos
        return loss

loss_func = FocalLoss()
opt = optim.Adam(model.parameters(), lr=0.001)

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=15)

# 현재 lr 계산
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# batch당 loss 계산
def loss_batch(loss_func, loc_preds, loc_targets, cls_preds, cls_targets, opt=None):
    loss_b = loss_func(loc_preds, loc_targets, cls_preds, cls_targets)
    
    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    
    return loss_b.item()

# epoch당 loss 계산
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    len_data = len(dataset_dl.dataset)

    for img, loc_targets, cls_targets in dataset_dl:
        img, loc_targets, cls_targets = img.to(device), loc_targets.to(device), cls_targets.to(device)
        loc_preds, cls_preds = model(img)

        loss_b = loss_batch(loss_func, loc_preds, loc_targets, cls_preds, cls_targets, opt)
        
        running_loss += loss_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    return loss

# 학습을 시작하는 함수
def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    opt=params['optimizer']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    sanity_check=params['sanity_check']
    lr_scheduler=params['lr_scheduler']
    path2weights=params['path2weights']

    loss_history = {'train': [], 'val': []}

    best_loss = float('inf')
    torch.save(model.state_dict(),path2weights)
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr = {}'.format(epoch, num_epochs-1, current_lr))

        model.train()
        train_loss = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),path2weights)
            print('Copied best model weights!')

        lr_scheduler.step(val_loss)

        if current_lr != get_lr(opt):
            print('Loading best model weights')
            model.load_state_dict(torch.load(path2weight))

        print('train loss: %.6f, val loss: %.6f, time: %.4f min' %(train_loss, val_loss, (time.time()-start_time)/60))

    model.load_state_dict(torch.load(path2weight))
    return model, loss_history

# train 파라미터 정의
params_train = {
    'num_epochs':100,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_dl,
    'val_dl':val_dl,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
}

# 가중치 저장할 폴더 생성
import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSerror:
        print('Error')
createFolder('./models')

model=RetinaNet().to(device)
model, loss_hist = train_val(model, params_train)

num_epochs = params_train['num_epochs']

# Plot train-val loss
plt.title('Train-Val Loss')
plt.plot(range(1, num_epochs+1), loss_hist['train'], label='train')
plt.plot(range(1, num_epochs+1), loss_hist['val'], label='val')
plt.ylabel('Loss')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()

model = RetinaNet().to(device)
model.load_state_dict(torch.load('/content/models/weights.pt'))
model.eval()

# test set trainforms 적용
IMAGE_SIZE = 600
scale = 1.0

test_transforms = A.Compose([
                    A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
                    A.PadIfNeeded(min_height=int(IMAGE_SIZE*scale), min_width=int(IMAGE_SIZE*scale),border_mode=cv2.BORDER_CONSTANT),
                    ToTensor()
                    ])

# test 이미지 불러오기
img = Image.open('./data/test.jpg')
w = h = 600
img = np.array(img.convert('RGB'))
img = test_transforms(image=img)
img = img['image']

x = img.unsqueeze(0).to(device) # [batch, H, W, 3]
loc_preds, cls_preds = model(x)

encoder = DataEncoder()
loc_preds, cls_preds = loc_preds.to('cpu'), cls_preds.to('cpu')

# nms 수행 및 출력 값을 바운딩박스 형태로 받아오기
boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w,h))

# 이미지 출력
img = transforms.ToPILImage()(img)
draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
plt.imshow(np.array(img))
##########################

## Task 6: Model

# input_ = Input(shape=(144, 144, 3), name='image')
# input_ = torch.randn(144, 7, 7).to(device)

# x = input_

# for i in range(0, 5):
#     n_filters = 2**(4 + i)
#     x = nn.Conv2d(3, n_filter)(x)
#     x = nn.ReLU()(x)
#     x = nn.BatchNorm2d(n_filters)(x)
#     x = nn.MaxPool2d(3, stride=2, padding=1)(x)

# x = torch.Flatten()(x)
# x = nn.Linear(256)(x)
# x = nn.ReLU()(x)

# #########################################
# def classOut(x):
#     x = nn.Linear(9, name='class_out')(x)
#     return nn.Softmax(x)
# #########################################

# # class_out = Dense(9, activation='softmax', name='class_out')(x)
# class_out = classOut(x)
# box_out = nn.Linear(2, name='box_out')(x)

# # model = tf.keras.models.Model(input_, [class_out, box_out])
# model = nn.Module(input_, [class_out, box_out])
# model.summary()

# # BottleNeck of ResNet
# class Bottleneck(nn.Module):
#     expand = 4

#     def __init__(self, in_channels, inner_channels, stride=1):
#         super().__init__()

#         self.conv1 = nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(inner_channels)
#         self.conv2 = nn.Conv2d(inner_channels, inner_channels, 3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(inner_channels)
#         self.conv3 = nn.Conv2d(inner_channels, inner_channels*self.expand, 1, stride=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(inner_channels*self.expand)
#         self.relu = nn.ReLU()

#         self.downsample = nn.Sequential()

#         if stride != 1 or in_channels != inner_channels*self.expand:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, inner_channels*self.expand, 1, stride=stride, bias=False),
#                 nn.BatchNorm2d(inner_channels*self.expand)
#             )

#         self.relu = nn.ReLU()

#     def forward(self, x):
#         output = self.relu(self.bn1(self.conv1(x)))
#         output = self.relu(self.bn2(self.conv2(output)))
#         output = self.bn3(self.conv3(output))
#         output = self.relu(output + self.downsample(x))

#         return output

# x = torch.randn(1, 56,13,13).to(device)
# net = Bottleneck(x.size(1), x.size(1)).to(device)
# output = net(x)
# print(output.size())

# ## Task 7: Custom Metric: IoU

# class IoU(tf.keras.metrics.Metric):
#     def __init__(self, **kwargs):
#         super(IoU, self).__init__(**kwargs)

#         self.iou = self.add_weight(name='iou', initializer='zeros')
#         self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
#         self.num_ex = self.add_weight(name='num_ex', initializer='zeros')
    
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         def get_box(y):
#             rows, cols = y[:, 0], y[:, 1]
#             rows, cols = rows * 144, cols * 144
#             y1, y2 = rows, rows + 52
#             x1, x2 = cols, cols + 52
#             return x1, y1, x2, y2
        
#         def get_area(x1, y1, x2, y2):
#             return tf.math.abs(x2 - x1) * tf.math.abs(y2 - y1)
        
#         gt_x1, gt_y1, gt_x2, gt_y2 = get_box(y_true)
#         p_x1, p_y1, p_x2, p_y2 = get_box(y_pred)

#         i_x1 = tf.maximum(gt_x1, p_x1)
#         i_y1 = tf.maximum(gt_y1, p_y1)
#         i_x2 = tf.minimum(gt_x2, p_x2)
#         i_y2 = tf.minimum(gt_y2, p_y2)

#         i_area = get_area(i_x1, i_y1, i_x2, i_y2)
#         u_area = get_area(gt_x1, gt_y1, gt_x2, gt_y2) + get_area(p_x1, p_y1, p_x2, p_y2) - i_area

#         iou = tf.math.divide(i_area, u_area)
#         self.num_ex.assign_add(1)
#         self.total_iou.assign_add(tf.reduce_mean(iou))
#         self.iou = tf.math.divide(self.total_iou, self.num_ex)
    
#     def result(self):
#         return self.iou
    
#     def reset_state(self):
#         self.iou = self.add_weight(name='iou', initializer='zeros')
#         self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
#         self.num_ex = self.add_weight(name='num_ex', initializer='zeros')
