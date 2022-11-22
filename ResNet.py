import torch.nn as nn
import torch



__all__ = ['resnet_O', 'resnet_F', 'resnet_N', 'resnet_B', 'resnet_C', 'resnet_P', 'resnet_K']


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, filter_size=3, skip_size=1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=filter_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=skip_size, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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

    def __init__(self, in_planes, planes, filter_size=3, skip_size=1, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=filter_size, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu3 = nn.ReLU(inplace=True)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=skip_size, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
    def __init__(self, in_planes, block, num_blocks, pool_size, filter_size=3, skip_size=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.layers = []
        out_planes = 0
        for idx, num_block in enumerate(num_blocks):
            if idx == 0:
                layer = self._make_layer(block, in_planes, num_block, filter_size, skip_size, stride=1)
                self.layers.append(layer)
            else:
                layer = self._make_layer(block, in_planes * pow(2, idx), num_block, filter_size, skip_size, stride=2)
                self.layers.append(layer)
                out_planes = in_planes * pow(2, idx)
        for i,layer in enumerate(self.layers):
            self.add_module('layer{}'.format(i+1),layer)
        # self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.pool = nn.AvgPool2d(pool_size)
        self.classifier = nn.Linear(out_planes*block.expansion, num_classes)
        

    def _make_layer(self, block, planes, num_blocks, filter_size, skip_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, filter_size, skip_size, stride))
            self.in_planes = planes * block.expansion

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
        
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)


# def resnet18():
#     return ResNet(64, BasicBlock, [2, 2, 2, 2], 4)

# def resnet34():
#     return ResNet(64, BasicBlock, [3, 4, 6, 3], 4)

# def resnet50():
#     return ResNet(64, Bottleneck, [3, 4, 6, 3], 8)

# def resnet20():
#     return ResNet(16, BasicBlock, [3, 3, 3])

# def resnet18_16():
#     return ResNet(16, BasicBlock, [2, 2, 2, 2], 4)

# def resnet32():
#     return ResNet(64, BasicBlock, [5, 5, 5])


# def resnet44():
#     return ResNet(BasicBlock, [7, 7, 7])


# def resnet56():
#     return ResNet(BasicBlock, [9, 9, 9])


# def resnet110():
#     return ResNet(BasicBlock, [18, 18, 18])


# def resnet1202():
#     return ResNet(BasicBlock, [200, 200, 200])

def resnet_O():
    return ResNet(64, Bottleneck, [3, 4, 6, 3], 4)

def resnet_F():
    print('wrong')
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
    print('wrong')
    # return ResNet(16, BasicBlock, [2, 2, 2, 2], 3, 3, 3)



# def resnet_


# AdaptiveAvgPool2d((1, 1))

if __name__ == "__main__":
    net=resnet_K().cuda()
    print(net)
    y = net(torch.randn(1, 3, 32, 32).cuda())
    print(y.size())

    # resnet50
    # ResNet(64, Bottleneck, [3, 4, 6, 3], 8)




    # for net_name in __all__:
    #     if net_name.startswith('resnet'):
    #         print(net_name)
    #         test(globals()[net_name]())
    #         print()


