# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #first convolutional block
        self.conv1 = nn.Conv1d(23, 128, kernel_size=51, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        #first pooling
        self.pool1 = nn.MaxPool1d(kernel_size=4, padding=0)

        #second convolutional block
        self.conv2 = nn.Conv1d(128, 128, kernel_size=51, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        #second pooling
        self.pool2 = nn.MaxPool1d(kernel_size=4, padding=0)

        #third convolutional block
        self.conv3 = nn.Conv1d(128, 128, kernel_size=51, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        #second pooling
        self.pool3 = nn.MaxPool1d(kernel_size=4, padding=0)

        self.fc1 =nn.Conv1d(128, 100, kernel_size=16, padding=0)
        self.fc2 = nn.Conv1d(100, 2, kernel_size=1, padding=0)

        self.m = nn.ReflectionPad2d((25, 25, 0, 0))

    def forward(self, x):
        out = torch.unsqueeze(x, dim=1)
        out = self.m(out)
        out = torch.squeeze(out, dim =1) #remove dim
        #first conv block
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.pool1(out) 

        #second conv block
        out = torch.unsqueeze(out, dim=1)#add dim of 1
        out = self.m(out)
        out = torch.squeeze(out, dim=1) #remove dim
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.pool2(out) 

        #third conv block
        out = torch.unsqueeze(out, dim=1)
        out = self.m(out)
        out = torch.squeeze(out, dim =1) #remove dim
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.pool3(out) 

        out = self.fc1(out)
        out = self.fc2(out)

        out = out.transpose(0,1) #nxbxt
        n, b, t = out.size()
        out = out.contiguous()
        out = out.view(-1, t*b) #nx(t*b)
        out = out.t() #(t*b)xn
        return out



