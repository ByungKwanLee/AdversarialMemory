import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

# Custom package
from model.memory import OrthogonalEmbeddedMemory

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, in_channels, num_classes, widen_factor, dropRate, mean, std):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        self.n = (depth - 4) / 6
        self.block = BasicBlock

        # Add parameters
        self.in_channels = in_channels
        self.mean = mean
        self.std = std

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(self.in_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(self.n, nChannels[0], nChannels[1], self.block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(self.n, nChannels[1], nChannels[2], self.block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(self.n, nChannels[2], nChannels[3], self.block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = (x-self.mean) / self.std
        out = self.conv1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class WideResNet_Plain(WideResNet):

    def __init__(self, depth, in_channels, num_classes, widen_factor, dropRate, mean, std):
        super(WideResNet_Plain, self).__init__(depth, in_channels, num_classes, widen_factor, dropRate, mean, std)
        self.initialize()

    def get_inference(self, x_adv):
        logit_adv   = self(x_adv)
        return logit_adv

# YEAR: 21.01.29
# ORTHOGONAL EMBEDDED MEMORY: TESTING
class WideResNet_OEM(WideResNet):

    def __init__(self, depth, in_channels, num_classes, isinitialize, widen_factor, dropRate, mean, std):
        super(WideResNet_OEM, self).__init__(depth, in_channels, num_classes, widen_factor, dropRate, mean, std)
        self.initialize()
        self.load_memory(isinitialize)

    def load_memory(self, isinitialize):
        self.OEM = OrthogonalEmbeddedMemory(memory_shape=torch.Size([512, 8, 8]), isinitialize=isinitialize).cuda()

    def forward(self, x, pop=False, intermediate_propagate=False):

        if intermediate_propagate:
            # propagation from intermediate layer
            out = x
        else:
            out = (x-self.mean) / self.std
            out = self.conv1(out)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)

            if pop:
                return out

            out = self.OEM(out)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


    def get_training_resource(self, x_clean, x_adv):
        assert (not self.training) and self.OEM.training
        feat_adv   = self(x_adv, pop=True)
        feat_clean = self(x_clean, pop=True)
        feat_memory,  memory_loss   = self.OEM(feat_adv.detach(), train=True)
        feat_memory_, memory_loss_ = self.OEM(feat_clean.detach(), train=True)
        logit = self(feat_memory, intermediate_propagate=True)
        logit_= self(feat_memory_, intermediate_propagate=True)
        return logit, logit_, memory_loss, memory_loss_

    def get_inference(self, x_adv):
        feat_adv   = self(x_adv, pop=True)
        feat_memory = self.OEM(feat_adv.detach())
        logit = self(feat_memory, intermediate_propagate=True)
        return logit

    def get_capturing_feature(self, x_adv, img_name):
        feat_adv = self(x_adv, pop=True)
        feat_memory = self.OEM(feat_adv.detach())
        attrib_adv = feat_adv[0].square().sum(dim=0)
        attrib_memory = feat_memory[0].square().sum(dim=0)
        save_image(attrib_adv/attrib_adv.max(), img_name+'_attrib_before_memory'+'.png')
        save_image(attrib_memory/attrib_memory.max(), img_name+'_attrib_after_memory'+'.png')


