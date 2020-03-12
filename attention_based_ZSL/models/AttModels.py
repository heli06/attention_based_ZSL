import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models



class AttEncoder(nn.Module):
    def __init__(self,args):
        super(AttEncoder,self).__init__()
        self.fc1 = nn.Linear(args.att_DIM,args.att_hidDIM)
        self.fc2 = nn.Linear(args.att_hidDIM, args.out_DIM)
        nn.init.xavier_uniform(self.fc1.weight.data)
        nn.init.xavier_uniform(self.fc2.weight.data)
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        return x