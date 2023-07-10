import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .prune import PruningModule, MaskedLinear

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5,5))
        self.fc1 = linear(120, 84)
        self.fc2 = linear(84, 10)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv3
        x = self.conv3(x)
        x = F.relu(x)

        # Fully-connected
        x = x.view(-1, 120)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG(PruningModule):
    def __init__(self, mask=False, in_channels=3, num_classes=10):
        super(VGG, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.in_channels = in_channels
        self.conv = self._make_layer(vgg16)
        self.fc = nn.Sequential(
			      linear(512*1*1, 4096),
			      nn.ReLU(),
			      nn.Dropout(p=0.5),
			      linear(4096, 4096),
			      nn.ReLU(),
			      nn.Dropout(p=0.5),
			      linear(4096, num_classes)
			  )

    def forward(self, x):
        x = self.conv(x)
        #print(x.shape)
        x = x.reshape(x.shape[0], -1)
        #print(x.shape)
        x = self.fc(x)
        return x

    def _make_layer(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                    out_channels = x
                    layers += [
						              nn.Conv2d(
							            in_channels=in_channels,
							            out_channels=out_channels,
							            kernel_size=(3, 3),
							            stride=(1, 1),
							            padding=(1, 1),
					              	),
					      	        nn.BatchNorm2d(x),
						              nn.ReLU(),
					          ]
                    in_channels = x
            elif x == "M":
                    layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers) 

#denseNet121
class Bottleneck(PruningModule):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(PruningModule):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(PruningModule):
    def __init__(self, block=Bottleneck, nblocks = [6,12,24,16], growth_rate=32, reduction=0.5, num_classes=10, mask=False):
        super(DenseNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)
