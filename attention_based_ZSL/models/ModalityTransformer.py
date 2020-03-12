import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models

class ModalityTransformer(nn.Module):
    def __init__(self, args):
        super(ModalityTransformer, self).__init__()
        self.fc = nn.Linear(args.out_DIM, args.out_DIM)
        self.conv1 = nn.Conv1d(args.out_DIM, args.out_DIM, 1)
        self.bn1 = nn.BatchNorm1d(args.out_DIM, momentum=0.5)
        self.conv2 = nn.Conv1d(args.out_DIM, args.out_DIM, 1)
        self.bn2 = nn.BatchNorm1d(args.out_DIM, momentum=0.5)
        nn.init.xavier_uniform(self.fc.weight.data)
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
    def forward(self, input):
        x = self.fc(input)
        out = self.conv1(x.unsqueeze(2)).squeeze(2)
        out = F.relu(out)
        out = self.bn1(out)
        out = self.conv2(out.unsqueeze(2)).squeeze(2)
        out = F.relu(out)
        out = self.bn2(out)
        out += x
        return out