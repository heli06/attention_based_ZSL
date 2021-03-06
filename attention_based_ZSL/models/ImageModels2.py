import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models
from . import Attention

"""
class Resnet101(imagemodels.ResNet):
    def __init__(self, pretrained=True):
        super(Resnet50, self).__init__(imagemodels.resnet.Bottleneck, [3, 4, 23, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet101']))

    def forward(self, x):
        x = nn.functional.interpolate(x,size=(244, 244), mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature = x
        x = self.maxpool(x)
        return feature,x
"""


class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        model = models.resnet101(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 1024)

        self.conv_map = nn.Conv2d(1024, 128, 1, stride=1)

    def init_trainable_weights(self):
        initrange = 0.1
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, attr, args):
        batch_size = x.size(0)
        x = nn.functional.interpolate(x, size=(244, 244), mode='bilinear', align_corners=False)
        # 1, 64, 122, 122。这里的1是batch_size
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 1, 64, 61, 61
        x = self.maxpool(x)
        # 1, 256, 61, 61
        x = self.layer1(x)
        # 1, 512, 31, 31
        x = self.layer2(x)
        # 1, 1024, 16, 16
        x = self.layer3(x)

        # 1, 128, 16, 16
        x_map = self.conv_map(x)
        # 1, 128, 256
        x_map = x_map.view(x_map.size(0), x_map.size(1), -1)
        # sim = Attention.score_att2img(x_map, attr, args)
        weiContext, attn = Attention.func_attention(x_map.permute(0, 2, 1), attr, args)
        print(weiContext.size())

        # 1, 2048, 8, 8
        x = self.layer4(x)
        # 1, 2048, 1, 1
        x = self.avgpool(x)
        # 1, 2048
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return nn.functional.normalize(x, p=2, dim=1), None