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
        self.fc2 = nn.Linear(args.att_hidDIM, args.att_outDIM)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        return nn.functional.normalize(x, p=2, dim=1)

class AttDecoder(nn.Module):
    def __init__(self,args):
        super(AttDecoder,self).__init__()
        self.fc1 = nn.Linear(args.out_DIM,args.att_hidDIM)
        self.fc2 = nn.Linear(args.att_hidDIM, args.att_DIM)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

class AttProj(nn.Module):
    def __init__(self,args):
        super(AttProj,self).__init__()
        self.fc1 = nn.Linear(args.att_DIM,args.att_mapHidDIM)
        self.fc2 = nn.Linear(args.att_mapHidDIM, args.att_mapOutDIM)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        return nn.functional.normalize(x, p=2, dim=1)


class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.gru = nn.GRU(1, hidden_size=args.attn_embedDIM // 2, num_layers=2, batch_first=True, bidirectional=True)
    def forward(self, input):
        self.gru.flatten_parameters()
        x, hn = self.gru(input, None)
        return x
