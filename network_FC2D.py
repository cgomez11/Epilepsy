# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #first convolutional block
        n_filters = 128
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=(23,3), padding=(0,1))
        #asymmetric kernel:  self.conv1 = nn.Conv2d(1, 128, kernel_size=(23,3), padding=11)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nn.ReLU()
        #first pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(1,4), padding=0)

        #second convolutional block
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=(1,3), padding=(0,1))
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nn.ReLU()
        #second pooling
        self.pool2 = nn.MaxPool2d(kernel_size=(1,4), padding=0)

        #third convolutional block
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=(1,3), padding=(0,1))
        self.bn3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU()
        #third pooling
        self.pool3 = nn.MaxPool2d(kernel_size=(1,4), padding=0)

        self.fc1 =nn.Conv2d(n_filters, 100, kernel_size=(1,16), padding=0)
        self.fc2 = nn.Conv2d(100, 2, kernel_size=(1,1), padding=0)

    def forward(self, x):
        #print('first data=', x.data)
        #Bx1x23x1024
        #first conv block
        out = self.relu1(self.bn1(self.conv1(x))) #Bxn_filx1x1024
        out = self.pool1(out) #Bxn_filx1x256

        out = self.relu2(self.bn2(self.conv2(out))) #Bxn_filx1x256
        out = self.pool2(out) #Bxn_filx1x64

        out = self.relu3(self.bn3(self.conv3(out))) #Bxn_filx1x64
        out = self.pool3(out) #Bxn_filx1x16

        out = self.fc1(out) #Bx100x1xt
        out =self.fc2(out) #Bx2x1xt

        out = torch.squeeze(out,2) #Bx2xt

        out = out.transpose(0,1) #2xBxt
        n, b, t = out.size()
        out = out.contiguous()
        out = out.view(-1, t*b) #nx(t*b)
        out = out.t() #(t*b)xn

        return out



