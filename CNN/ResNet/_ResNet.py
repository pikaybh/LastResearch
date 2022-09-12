import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import imageio
import argparse

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# utils
import numpy as np
from torchsummary import summary

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--layer', '-L', default="1-1-1-1")
args = parser.parse_args()

_layers = args.layer.split("-")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

original = imageio.imread('../data/test.jpg')

# from [H, W, C] to [C, H, W]
transposed_image = original.transpose((2, 0, 1))
# add batch dim
transposed_image = np.expand_dims(transposed_image, 0)

image_d = torch.FloatTensor(transposed_image)

# Convolution Layer
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    r"""
    3x3 convolution with padding
    - in_planes: in_channels
    - out_channels: out_channels
    - bias=False: BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Blocks
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        r"""
         - inplanes: input channel size
         - planes: output channel size
         - groups, base_width: ResNext나 Wide ResNet의 경우 사용
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Basic Block의 구조
        self.conv1 = conv3x3(inplanes, planes, stride)  # conv1에서 downsample
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # short connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # identity mapping시 identity mapping후 ReLU를 적용합니다.
        # 그 이유는, ReLU를 통과하면 양의 값만 남기 때문에 Residual의 의미가 제대로 유지되지 않기 때문입니다.
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4  # 블록 내에서 차원을 증가시키는 3번째 conv layer에서의 확장계수

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # ResNext나 WideResNet의 경우 사용
        width = int(planes * (base_width / 64.)) * groups

        # Bottleneck Block의 구조
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # conv2에서 downsample
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # 1x1 convolution layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3x3 convolution layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 1x1 convolution layer
        out = self.conv3(out)
        out = self.bn3(out)
        # skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, b=1):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        # default values
        self.inplanes = 64  # input feature map
        self.dilation = 1
        # stride를 dilation으로 대체할지 선택
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        r"""
        - 처음 입력에 적용되는 self.conv1과 self.bn1, self.relu는 모든 ResNet에서 동일 
        - 3: 입력으로 RGB 이미지를 사용하기 때문에 convolution layer에 들어오는 input의 channel 수는 3
        """
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        r"""
        - 아래부터 block 형태와 갯수가 ResNet층마다 변화
        - self.layer1 ~ 4: 필터의 개수는 각 block들을 거치면서 증가(64->128->256->512)
        - self.avgpool: 모든 block을 거친 후에는 Adaptive AvgPool2d를 적용하여 (n, 512, 1, 1)의 텐서로
        - self.fc: 이후 fc layer를 연결
        """
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,  # 여기서부터 downsampling적용
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.b = b

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        r"""
        convolution layer 생성 함수
        - block: block종류 지정
        - planes: feature map size (input shape)
        - blocks: layers[0]와 같이, 해당 블록이 몇개 생성돼야하는지, 블록의 갯수 (layer 반복해서 쌓는 개수)
        - stride와 dilate은 고정
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # the number of filters is doubled: self.inplanes와 planes 사이즈를 맞춰주기 위한 projection shortcut
        # the feature map size is halved: stride=2로 downsampling
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 블록 내 시작 layer, downsampling 필요
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion  # inplanes 업데이트
        # 동일 블록 반복
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, b):
        # See note [TorchScript super()]
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3)

        x5 = self.layer1(x4)
        x6 = self.layer2(x5)
        x7 = self.layer3(x6)
        x8 = self.layer4(x7)

        x9 = self.avgpool(x8)
        x10 = torch.flatten(x9, 1)
        x11 = self.fc(x10)
        out = x11

        x = x.permute(0, 1, 2, 3)[0, 1]
        x1 = x1.permute(0, 1, 2, 3)[0, 1]
        x2 = x2.permute(0, 1, 2, 3)[0, 1]
        x3 = x3.permute(0, 1, 2, 3)[0, 1]
        x4 = x4.permute(0, 1, 2, 3)[0, 1]
        x5 = x5.permute(0, 1, 2, 3)[0, 1]
        x6 = x6.permute(0, 1, 2, 3)[0, 1]
        x7 = x7.permute(0, 1, 2, 3)[0, 1]
        x8 = x8.permute(0, 1, 2, 3)[0, 1]
        x9 = x9.permute(0, 2, 1, 3)[0]
        x10 = x10.permute(1, 0)
        x11 = x11.permute(1, 0)

        x = x.detach().numpy()
        x1 = x1.detach().numpy()
        x2 = x2.detach().numpy()
        x3 = x3.detach().numpy()
        x4 = x4.detach().numpy()
        x5 = x5.detach().numpy()
        x6 = x6.detach().numpy()
        x7 = x7.detach().numpy()
        x8 = x8.detach().numpy()
        x9 = x9.detach().numpy()
        x10 = x10.detach().numpy()
        x11 = x11.detach().numpy()

        number_of_pic = (5, 5)
        plt.figure(figsize=(10, 10))
        plt.subplot(number_of_pic[0], number_of_pic[1], 1)
        plt.title("test image")
        plt.imshow(original)
        plt.subplot(number_of_pic[0], number_of_pic[1], 2)
        plt.title("x")
        plt.imshow(x)
        plt.subplot(number_of_pic[0], number_of_pic[1], 3)
        plt.title("conv1")
        plt.imshow(x1)
        plt.subplot(number_of_pic[0], number_of_pic[1], 4)
        plt.title("bn1")
        plt.imshow(x2)
        plt.subplot(number_of_pic[0], number_of_pic[1], 5)
        plt.title("relu")
        plt.imshow(x3)
        plt.subplot(number_of_pic[0], number_of_pic[1], 6)
        plt.title("max pool")
        plt.imshow(x4)
        plt.subplot(number_of_pic[0], number_of_pic[1], 7)
        plt.title("layer1")
        plt.imshow(x5)
        plt.subplot(number_of_pic[0], number_of_pic[1], 8)
        plt.title("layer2")
        plt.imshow(x6)
        plt.subplot(number_of_pic[0], number_of_pic[1], 9)
        plt.title("layer3")
        plt.imshow(x7)
        plt.subplot(number_of_pic[0], number_of_pic[1], 10)
        plt.title("layer4")
        plt.imshow(x8)
        plt.subplot(number_of_pic[0], 1, 3)
        plt.title("avg pool")
        plt.plot(x9[0])
        plt.subplot(number_of_pic[0], 1, 4)
        plt.title("flatten")
        plt.plot(x10)
        plt.subplot(number_of_pic[0], 1, 5)
        plt.title("fc")
        plt.plot(x11)
        plt.show()
        # plt.savefig('./graphs/epoch={}.png'.format(str(self.b)), dpi=300)
        self.b += 100

        return out

    def forward(self, x, b):
        return self._forward_impl(x, b)

def train(model, device, train_loader, optimizer, criterion, epoch, steps_per_epoch=20):
    # Log gradients and model parameters
    # wandb.watch(model)

    # loop over the data iterator, and feed the inputs to the network and adjust the weights.
    for batch_idx, (data, target) in enumerate(train_loader, start=0):
    
        # ...
        
        acc = round((train_correct / train_total) * 100, 2)
        # Log metrics to visualize performance
        # wandb.log({'Train Loss': train_loss/train_total, 'Train Accuracy': acc})

        if batch_idx%100 ==0:
            ResNet_model(image_d, b=1)



if __name__ == "__main__":
    # train_loader = DataLoader("../data", batch_size=128, shuffle=True, num_workers=4)
    ResNet_model = ResNet(block=BasicBlock, layers=[int(_layers[0]), int(_layers[1]), int(_layers[2]), int(_layers[3])]).to(device)
    ResNet_model(image_d, b=1)
    # summary(ResNet_model, (3, 28, 28))

    # 손실 함수와 optimizer를 생성합니다. 이 이상한 모델을 순수한 확률적 경사하강법(SGD; Stochastic Gradient Descent)으로
    # 학습하는 것은 어려우므로, 모멘텀(momentum)을 사용합니다.
    # criterion = torch.nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.SGD(ResNet_model.parameters(), lr=1e-8, momentum=0.9)

    # train(ResNet_model, device=device, train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=1000)

        # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
