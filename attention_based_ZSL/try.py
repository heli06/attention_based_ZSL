import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models import ImageModels, AttModels, ImageModels2

image_model = ImageModels.Resnet101()
input = torch.randn(1, 3, 244, 244)
output = image_model(input)
print(output.size())