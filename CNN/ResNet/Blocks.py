import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio

device = torch.device("cpu")

original = imageio.imread('test.jpg')

# from [H, W, C] to [C, H, W]
transposed_image = original.transpose((2, 0, 1))
# add batch dim
transposed_image = np.expand_dims(transposed_image, 0)

image_d = torch.FloatTensor(transposed_image)


# con
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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4 # 블록 내에서 차원을 증가시키는 3번째 conv layer에서의 확장계수

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
        self.conv2 = conv3x3(width, width, stride, groups, dilation) # conv2에서 downsample
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # 1x1 convolution layer
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        # 3x3 convolution layer
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        x6 = self.relu(x5)
        # 1x1 convolution layer
        x7 = self.conv3(x6)
        x8 = self.bn3(x7)
        # skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        x8 += identity
        x9 = self.relu(x8)
        out = x9

        x = x.permute(0, 1, 2, 3)[0]
        x1 = x1.permute(0, 1, 2, 3)[0]
        x2 = x2.permute(0, 1, 2, 3)[0]
        x3 = x3.permute(0, 1, 2, 3)[0]
        x4 = x4.permute(0, 1, 2, 3)[0]
        x5 = x5.permute(0, 1, 2, 3)[0]
        x6 = x6.permute(0, 1, 2, 3)[0]
        x7 = x7.permute(0, 1, 2, 3)[0]
        x8 = x8.permute(0, 1, 2, 3)[0]
        x9 = x9.permute(0, 1, 2, 3)[0]

        x = x.permute(0, 1, 2)[0]
        x1 = x1.permute(0, 1, 2)[0]
        x2 = x2.permute(0, 1, 2)[0]
        x3 = x3.permute(0, 1, 2)[0]
        x4 = x4.permute(0, 1, 2)[0]
        x5 = x5.permute(0, 1, 2)[0]
        x6 = x6.permute(0, 1, 2)[0]
        x7 = x7.permute(0, 1, 2)[0]
        x8 = x8.permute(0, 1, 2)[0]
        x9 = x9.permute(0, 1, 2)[0]

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

        numberOfPic = 10
        plt.figure(figsize=(8, 8))
        plt.subplot(numberOfPic, 1, 1)
        plt.title("identity")
        plt.imshow(x)
        plt.subplot(numberOfPic, 1, 2)
        plt.title("conv1")
        plt.imshow(x1)
        plt.subplot(numberOfPic, 1, 3)
        plt.title("bn1")
        plt.imshow(x2)
        plt.subplot(numberOfPic, 1, 4)
        plt.title("relu")
        plt.imshow(x3)
        plt.subplot(numberOfPic, 1, 5)
        plt.title("conv2")
        plt.imshow(x4)
        plt.subplot(numberOfPic, 1, 6)
        plt.title("bn2")
        plt.imshow(x5)
        plt.subplot(numberOfPic, 1, 7)
        plt.title("relu")
        plt.imshow(x6)
        plt.subplot(numberOfPic, 1, 8)
        plt.title("con3")
        plt.imshow(x7)
        plt.subplot(numberOfPic, 1, 9)
        plt.title("bn3")
        plt.imshow(x8)
        plt.subplot(numberOfPic, 1, 10)
        plt.title("relu")
        plt.imshow(x9)
        plt.show()

        return out


BottleNeck_model = Bottleneck(inplanes=3, planes=3).to(device)
image_g = transforms.Grayscale(num_output_channels=1)(image_d)
BottleNeck_model(image_d)
# summary(BottleNeck_model, (1, 28, 28))