# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #first convolutional block
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3,23), padding=(1,0))
        #asymmetric kernel:  self.conv1 = nn.Conv2d(1, 128, kernel_size=(23,3), padding=11)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        #first pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(4,1), padding=0)
        
        #second convolutional block
        self.conv2 = nn.Conv2d(1, 128, kernel_size=(3,128), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        #second pooling
        self.pool2 = nn.MaxPool2d(kernel_size=(4,1), padding=0)
        
        #third convolutional block
        self.conv3 = nn.Conv2d(1, 128, kernel_size=(3,128), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        #third pooling
        self.pool3 = nn.MaxPool2d(kernel_size=(4,1), padding=0)
        #feature-map size = 1x16
        
        #fully convolutional networks
        fc1 = nn.Conv2d(1, 100, kernel_size=(16,128), padding=0)
        
        fc2 = nn.Conv2d(1, 2, kernel_size=(1,100), padding=0)
        #self.relu5 = nn.ReLU()
        #self.relu6 = nn.ReLU()
        self.softmax = nn.Softmax()
     
        #self.fc1 = nn.Linear(128*6*6*6, 128)
        #self.relu13 = nn.ReLU()
    
        
    def forward(self, x):
        #print(x.data)
        #first conv block
        out = self.relu1(self.bn1(self.conv1(x)))
        ############### reshape #################
        out = torch.squeeze(out, 3)
        out = torch.transpose(out,1,2)
        out = torch.unsqueeze(out, 1)  #16x1x1024x128
        out = self.pool1(out)   #16x1x256x128
        #print(out.data)

        #second block   
        out = self.relu2(self.bn2(self.conv2(out)))
        ############### reshape #################
        out = torch.squeeze(out, 3)
        out = torch.transpose(out,1,2)
        out = torch.unsqueeze(out, 1)  #16x1x256x128
        out = self.pool2(out) #16x1x64x128

        #third conv block
        #print(out.data)
        out = self.relu3(self.bn3(self.conv3(out)))
        out = torch.squeeze(out, 3)
        out = torch.transpose(out,1,2)
        out = torch.unsqueeze(out, 1)  #16x1x64x128
        out = self.pool3(out)  #16x1x16x128

        #fully connected layers
        out = self.fc1(out)  #16x100x1x1 o 16x100xtx1
        out = torch.squeeze(out, 3)
        out = torch.transpose(out,1,2)
        out = torch.unsqueeze(out, 1)  #16x1xtx100

        out =self.fc2(out) #16x2xtx1
        out = torch.squeeze(out, 3)
        out = torch.transpose(out,1,2) #16xtxn
        #print(out.data)
        out = self.softmax(out)
        return out
