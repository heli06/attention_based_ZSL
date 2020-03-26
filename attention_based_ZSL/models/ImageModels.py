import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models

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
        
        self.fc1 = nn.Linear(2048,2048)
        self.fc2 = nn.Linear(2048,2048)

    def init_trainable_weights(self):
        initrange = 0.1   
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)

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
        # feature = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class ImgDecoder(nn.Module):
    def __init__(self,args):
        super(ImgDecoder,self).__init__()
        self.fc1 = nn.Linear(args.out_DIM,args.out_DIM)
        self.fc2 = nn.Linear(args.out_DIM, args.out_DIM)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.tanh(x)
        return x