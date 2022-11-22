'''
Jiayi Shang
11/15/2022

'''
import torch.nn as nn
import torch



__all__ = ['resnet_O', 'resnet_F', 'resnet_N', 'resnet_B', 'resnet_C', 'resnet_P', 'resnet_K']


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channals, channals, filter_size=3, skip_size=1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channals, channals, kernel_size=filter_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channals)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channals, channals, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channals)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channals != self.expansion*channals:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channals, self.expansion*channals, kernel_size=skip_size, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channals)
            )

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:
            x = self.downsample(x)

        y += self.shortcut(x)
        y = self.relu2(y)
        return y

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channals, channals, filter_size=3, skip_size=1, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channals, channals, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channals)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channals, channals, kernel_size=filter_size, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channals)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channals, channals * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channals * Bottleneck.expansion)
        self.relu3 = nn.ReLU(inplace=True)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channals != self.expansion*channals:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channals, self.expansion*channals, kernel_size=skip_size, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channals)
            )

    def forward(self, x):

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)

        y = self.conv3(y)
        y = self.bn3(y)

        if self.downsample is not None:
            x = self.downsample(x)

        y += self.shortcut(x)
        y = self.relu3(y)

        return y


class ResNet(nn.Module):
    def __init__(self, in_channals, block, num_blocks, pool_size, filter_size=3, skip_size=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channals = in_channals

        self.conv1 = nn.Conv2d(3, in_channals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channals)
        self.relu1 = nn.ReLU(inplace=True)
        self.layers = []
        out_planes = 0
        for idx, num_block in enumerate(num_blocks):
            if idx == 0:
                layer = self._make_layer(block, in_channals, num_block, filter_size, skip_size, stride=1)
                self.layers.append(layer)
            else:
                layer = self._make_layer(block, in_channals * pow(2, idx), num_block, filter_size, skip_size, stride=2)
                self.layers.append(layer)
                out_planes = in_channals * pow(2, idx)
        for i,layer in enumerate(self.layers):
            self.add_module('layer{}'.format(i+1),layer)
        self.pool = nn.AvgPool2d(pool_size)
        # self.pool = nn.MaxPool2d(pool_size)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(out_planes*block.expansion, num_classes)
        

    def _make_layer(self, block, channals, num_blocks, filter_size, skip_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channals, channals, filter_size, skip_size, stride))
            self.in_channals = channals * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        for layer in self.layers:
            # print(layer)
            y = layer(y)
        # print(y.size())
        y = self.pool(y)
        y = y.view(y.size(0),-1)
        y = self.classifier(y)
        return y
        


def resnet_O():
    return ResNet(64, Bottleneck, [3, 4, 6, 3], 4)

def resnet_F():
    pass
    # return ResNet(64, Bottleneck, [3, 4, 6, 3], 8)

def resnet_N():
    return ResNet(64, BasicBlock, [3, 4, 6], 8)

def resnet_B():
    return ResNet(64, BasicBlock, [2, 2, 2, 2], 4)

def resnet_C():
    return ResNet(16, BasicBlock, [2, 2, 2, 2], 4)

def resnet_P():
    return ResNet(16, BasicBlock, [2, 2, 2, 2], 3)

def resnet_K():
    pass
    # return ResNet(16, BasicBlock, [2, 2, 2, 2], 3, 3, 3)





if __name__ == "__main__":
    net=resnet_B().cuda()
    print(net)
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total: ', total_num)
    print('Trainable: ', trainable_num)
    y = net(torch.randn(1, 3, 32, 32).cuda())
    print(y.size())


