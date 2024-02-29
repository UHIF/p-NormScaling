import torch
import torchvision.transforms as transforms
import torch as t
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from sklearn.model_selection import StratifiedShuffleSplit
from calibration import calibration_curve
from calibration import calibration_error
from calibration import pNorm_calibration


t.manual_seed(8)
np.random.seed(8)
t.cuda.manual_seed(5)
t.cuda.manual_seed_all(5)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset=tv.datasets.CIFAR100(
    root='C:/home/cy/data',
    train=True,
    download=True,
    transform=transform_train
)

labels = [label for _, label in trainset]

# Split train dataset into train and validation sets using stratified sampling
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_indices, val_indices = next(sss.split(trainset.data, labels))

train_dataset = torch.utils.data.Subset(trainset, train_indices)
val_dataset = torch.utils.data.Subset(trainset, val_indices)


val_loader=DataLoader(
    val_dataset,
    batch_size=100,
    shuffle=True,
    num_workers=0
)
trainloader=DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=0
)
testset=tv.datasets.CIFAR100(
    'C:/home/cy/data',
    train=False,
    download=True,
    transform=transform_test
)
testloader=DataLoader(
    testset,
    batch_size=100,
    shuffle=False,
    num_workers=0
)

class Net2(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,15,kernel_size=(3,3),stride=(1,1))
        self.conv2=nn.Conv2d(15,75,kernel_size=(4,4),stride=(1,1))
        self.conv3=nn.Conv2d(75,175,kernel_size=(3,3),stride=(1,1))
        self.fc1=nn.Linear(700,200)
        self.fc2=nn.Linear(200,120)
        self.fc3=nn.Linear(120,84)
        self.fc4=nn.Linear(84,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=F.max_pool2d(F.relu(self.conv3(x)),2)

        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        #self.weightnorm=weight_norm(nn.Linear(512 * block.expansion, num_classes))
        #self.layernorm=nn.LayerNorm([100,512],elementwise_affine=False)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        #K=torch.sum(feature*feature,axis=1,keepdims=True)
        #out = feature/torch.sqrt(K)
        out= self.linear(feature)
        #out=self.layernorm(feature)
        #out = self.weightnorm(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 16
            nn.Linear(4096, num_classes),
        )
        # self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        #        print(out.shape)
        out = out.view(out.size(0), -1)
        #        print(out.shape)
        out = self.classifier(out)
        #        print(out.shape)
        return out
def conv_3x3_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU6(inplace=True),
    )


def dws_conv_3x3_bn(in_channels, out_channels, dw_stride):
    """
    Depthwise Separable Convolution
    :param in_channels: depthwise conv input channels
    :param out_channels: Separable conv output channels
    :param dw_stride: depthwise conv stride
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=in_channels,
                  kernel_size=3,
                  stride=dw_stride,
                  padding=1,
                  groups=in_channels,
                  bias=False,
                  ),
        nn.BatchNorm2d(num_features=in_channels),
        nn.ReLU6(inplace=True),

        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=1,
                  stride=1,
                  bias=False,
                  ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU6(inplace=True),
    )


class MobileNetV1(nn.Module):
    """
    mobilenet V1 implementation. Modify a bit for CIFAR10
    """
    def __init__(self, num_classes=100  ):
        super(MobileNetV1, self).__init__()
        layers = [conv_3x3_bn(in_channels=3, out_channels=32, stride=1)]  # change stride 2->1 for cifar10
        dws_conv_config = [
            # num, in_channels, out_channels, stride
            [1, 32, 64, 1],
            [1, 64, 128, 1],  # change stride 2->1 for cifar10
            [1, 128, 128, 1],
            [1, 128, 256, 2],
            [1, 256, 256, 1],
            [1, 256, 512, 2],
            [5, 512, 512, 1],
            [1, 512, 1024, 2],
            [1, 1024, 1024, 1]
        ]
        for num, in_channels, out_channels, dw_stride in dws_conv_config:
            for i in range(num):
                layers.append(dws_conv_3x3_bn(in_channels, out_channels, dw_stride))
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        # print(x.shape)
        y = self.layers(x)
        # print(y.shape)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidualBlock, self).__init__()
        assert stride == 1 or stride == 2
        self.stride = stride
        self.residual = self.stride == 1 and (in_channels == out_channels)
        expansion_channels = in_channels * expansion_factor

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=expansion_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=expansion_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=expansion_channels,
                      out_channels=expansion_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=expansion_channels,
                      bias=False),
            nn.BatchNorm2d(num_features=expansion_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=expansion_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        y = self.block(x)
        if self.residual:
            return y + x
        else:
            return y


class MobileNetV2(nn.Module):
    """
    mobilenet V2 implementation. modify a bit for CIFAR10
    """
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        layers = [conv_3x3_bn(in_channels=3, out_channels=32, stride=1)] # change stride 2->1 for cifar10
        in_channels = 32
        inverted_residual_block_config = [
            # expansion factor, out_channels, stride
            [1, 16, 1],

            [6, 24, 1],  # change stride 2->1 for cifar10
            [6, 24, 1],

            [6, 32, 2],
            [6, 32, 1],
            [6, 32, 1],

            [6, 64, 2],
            [6, 64, 1],
            [6, 64, 1],
            [6, 64, 1],

            [6, 96, 1],
            [6, 96, 1],
            [6, 96, 1],

            [6, 160, 2],
            [6, 160, 1],
            [6, 160, 1],

            [6, 320, 1],
        ]
        for expansion_factor, out_channels, stride in inverted_residual_block_config:
            layers.append(InvertedResidualBlock(in_channels, out_channels, stride, expansion_factor))
            in_channels = out_channels
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=320,
                      out_channels=1280,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=1280),
            # nn.ReLU6(inplace=True),
        ))
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1280, out_features=num_classes)
        )

    def forward(self, x):
        # print(x.shape)
        y = self.layers(x)
        # print(y.shape)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,200)
        self.fc2=nn.Linear(200,100)
        self.fc3=nn.Linear(100,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


model = torch.load('ResNet34_original.pkl')
print(model)  # 打印模型结构
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
p = 1
ece_best=1
p_best=1
temperature_best=1
bias_best=1
while p <= 3:
    temp_scale = pNorm_calibration.pNorm_calibration(model)
    temperature,bias=temp_scale.set_temperature(val_loader,p)
    ece, mse, adaece= calibration_error.calibration_error(model, testloader,temperature,bias,p)
    print(ece,mse,adaece)
    if ece_best < ece:
        p_best=p
        ece_best=ece
        temperature_best=temperature
        bias_best=bias
    p += 0.25

ece, mse, adaece= calibration_error.calibration_error(model, testloader,temperature_best,bias_best,p_best)
calibration_curve.calibration_curve(model, testloader,temperature_best,bias_best,p_best)




