import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models

class RelationNet(nn.Module):
    def __init__(self,args):
        super(RelationNet, self).__init__()
        input_dim = args.img_outDIM + args.att_outDIM
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 1)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x