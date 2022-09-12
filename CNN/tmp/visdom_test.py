import visdom
import torch

### MNIST 이미지 불러오기 ###
import torchvision.datasets as dsets
import torchvision

vis = visdom.Visdom() # 서버가 꺼져있으며 오류나기 때문에 반드시 실행되어야 함

# # Text
# vis.text("Hello, world!",env="main")

# # Image
# a=torch.randn(3,200,200)
# vis.image(a)

# Images
vis.images(torch.Tensor(3,3,28,28)) 

MNIST = dsets.MNIST(
    root="./data/MNIST_data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

data = MNIST.__getitem__(0)
print(data[0].shape)
vis.images(data[0], env="main")

### MNIST 이미지 여러 개 띄우기 ###
data_loader = torch.utils.data.DataLoader(
    dataset=MNIST,
    batch_size=32,
    shuffle=False
)

for num, value in enumerate(data_loader):
    value = value[0]
    print(value.shape)
    vis.images(value)
    break