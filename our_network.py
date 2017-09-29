# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #first convolutional block
        #self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        #asymmetric kernel:  
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(23,3), padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        #first pooling
        self.pool1 = nn.AvgPool2d(kernel_size=(1,4), padding=0)
        
        #second convolutional block
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        #second pooling
        self.pool2 = nn.AvgPool2d(kernel_size=(1,4), padding=0)
        
        #third convolutional block
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        #third pooling
        self.pool3 = nn.AvgPool2d(kernel_size=(1,4), padding=0)
        
        self.fc1 = nn.Linear(128, 2) #there are two classes?
        self.softmax = nn.Softmax()
     
        #self.fc1 = nn.Linear(128*6*6*6, 128)
        #self.relu13 = nn.ReLU()
    
        
    def forward(self, x):
        out = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        out = self.pool1(self.relu2(self.bn2(self.conv2(out))))
        out = self.pool3(self.relu3(self.bn3(self.conv3(out))))
        #out = torch.squeeze(F.avg_pool2d(out), (23,12))) #haria falta para que queden de 1x1?
        out = self.fc1(out.view(-1, 128))
        out = self.softmax(out) 
        return out
