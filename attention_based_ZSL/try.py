import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models import ImageModels, AttModels, ImageModels2
from models import RelationNet

image_model = ImageModels.Resnet101()
input = torch.randn(1, 3, 244, 244)
output = image_model(input)

relation_net = RelationNet.RelationNet()
input = torch.randn(64, 2360)
output = relation_net(input)
print(output)
index = int(torch.max(output[:,0], -1)[1])
print(index)