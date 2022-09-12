import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
import imageio

device = torch.device("cpu")

original = imageio.imread('test.jpg')

# summary(net_model, (3, 28, 28))

# from [H, W, C] to [C, H, W]
transposed_image = original.transpose((2, 0, 1))
# add batch dim
transposed_image = np.expand_dims(transposed_image, 0)

image_d = torch.FloatTensor(transposed_image)

plt.figure(figsize=(5, 5))
plt.subplot(1, 4, 1)
plt.title("original image")
plt.imshow(original)
plt.show()

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

# Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3405780, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
        # print("[x2 shape]: ", x2.shape)
        # x3 = x2.view(-1, 3200)
        # x3 = nn.Linear(3200, 320)
        # x3 = x2.view(-1, 320)
        x3 = x2.view(-1, 3405780)
        x4 = F.relu(self.fc1(x3))
        x5 = F.dropout(x4, training=self.training)
        x6 = self.fc2(x5)
        xF = x6

        x = x.permute(0, 1, 2, 3)[0]
        x = x.permute(0, 1, 2)[0]

        x1 = x1.permute(0, 1, 2, 3)[0]
        x1 = x1.permute(0, 1, 2)[0]

        x2 = x2.permute(0, 1, 2, 3)[0]
        x2 = x2.permute(0, 1, 2)[0]

        # x3 = x3.permute(0, 1, 2, 3)[0]
        # x4 = x4.permute(0, 1, 2, 3)[0]
        # x5 = x5.permute(0, 1, 2, 3)[0]
        # x6 = x6.permute(0, 1, 2, 3)[0]

        x1 = x1.detach().numpy()
        x2 = x2.detach().numpy()
        x3 = x3.detach().numpy()
        x4 = x4.detach().numpy()
        x5 = x5.detach().numpy()
        x6 = x6.detach().numpy()

        plt.figure(figsize=(10, 10))
        plt.subplot(6, 1, 1)
        plt.title("conv1")
        plt.imshow(x1)
        plt.subplot(6, 1, 2)
        plt.title("conv2")
        plt.imshow(x2)
        plt.subplot(6, 1, 3)
        plt.title("view")
        plt.imshow(x3)
        plt.subplot(6, 1, 4)
        plt.title("Linear")
        plt.imshow(x4)
        plt.subplot(6, 1, 5)
        plt.title("dropout")
        plt.imshow(x5)
        plt.subplot(6, 1, 6)
        plt.title("Linear")
        plt.imshow(x6)
        plt.show()

        return F.log_softmax(xF, dim=1)

net_model = Net().to(device)
net_model(image_d)
summary(net_model, (1, 28, 28))

# BB
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

        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)

        x4 = self.conv2(x3)
        x5 = self.bn2(x4)

        # short connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # identity mapping시 identity mapping후 ReLU를 적용합니다.
        # 그 이유는, ReLU를 통과하면 양의 값만 남기 때문에 Residual의 의미가 제대로 유지되지 않기 때문입니다.
        x5 += identity
        x6 = self.relu(x5)
        out = x6

        x = x.permute(0, 1, 2, 3)[0, 1]
        x1 = x1.permute(0, 1, 2, 3)[0, 1]
        x2 = x2.permute(0, 1, 2, 3)[0, 1]
        x3 = x3.permute(0, 1, 2, 3)[0, 1]
        x4 = x4.permute(0, 1, 2, 3)[0, 1]
        x5 = x5.permute(0, 1, 2, 3)[0, 1]
        x6 = x6.permute(0, 1, 2, 3)[0, 1]

        x = x.detach().numpy()
        x1 = x1.detach().numpy()
        x2 = x2.detach().numpy()
        x3 = x3.detach().numpy()
        x4 = x4.detach().numpy()
        x5 = x5.detach().numpy()
        x6 = x6.detach().numpy()

        plt.figure(figsize=(10, 10))
        plt.subplot(7, 1, 1)
        plt.title("identity")
        plt.imshow(x)
        plt.subplot(7, 1, 2)
        plt.title("conv1")
        plt.imshow(x1)
        plt.subplot(7, 1, 3)
        plt.title("bn1")
        plt.imshow(x2)
        plt.subplot(7, 1, 4)
        plt.title("relu")
        plt.imshow(x3)
        plt.subplot(7, 1, 5)
        plt.title("conv2")
        plt.imshow(x4)
        plt.subplot(7, 1, 6)
        plt.title("bn2")
        plt.imshow(x5)
        plt.subplot(7, 1, 7)
        plt.title("relu")
        plt.imshow(x6)
        plt.show()

        return out

BB_model = BasicBlock(inplanes=3, planes=3).to(device)
BB_model(image_d)
summary(BB_model, (3, 28, 28))













