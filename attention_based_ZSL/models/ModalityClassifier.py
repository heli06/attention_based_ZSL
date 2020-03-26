import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models

class ModalityClassifier(nn.Module):
    def __init__(self, args):
        super(ModalityClassifier, self).__init__()
        self.fc1 = nn.Linear(args.out_DIM, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
        nn.init.xavier_uniform_(self.fc3.weight.data)
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x)
        return x