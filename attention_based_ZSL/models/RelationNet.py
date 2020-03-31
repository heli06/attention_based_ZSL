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
        self.fc1 = nn.Linear(input_dim, args.rel_hidDIM)
        self.fc2 = nn.Linear(args.rel_hidDIM, 2)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        return x