import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, m=0.5, s=64, easy_margin=False, embedding_size=512):
        super(ArcFaceLoss, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        # num_classes Total number of face classifications in the training set
        # emb_size eigenvector length
        nn.init.xavier_uniform_(self.weight)
        # Use uniform distribution to initialize weight

        self.easy_margin = easy_margin
        self.m = m
        #  0.5 in the formula
        self.s = s
        # radius 64 s in the formula

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # cos and sin
        self.th = math.cos(math.pi - self.m)
        # threshold, avoid theta + m >= pi
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda:0')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output