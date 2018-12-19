import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #first convolutional block
        H = 288
        n_filters = 128
        self.conv1 = nn.Conv2d(3, n_filters, kernel_size=(H,3), padding=(0,1))
        #asymmetric kernel:  self.conv1 = nn.Conv2d(1, 128, kernel_size=(23,3), padding=11)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nn.ReLU()
        #first pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), padding=0)

        #second convolutional block
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=(1,3), padding=(0,1))
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nn.ReLU()
        #second pooling
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2), padding=0)

        #third convolutional block
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=(1,3), padding=(0,1))
        self.bn3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU()
        #second pooling
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2), padding=0)

        self.fc1 =nn.Conv2d(n_filters, 100, kernel_size=(1,36), stride=(1,36),padding=0)
        self.fc2 = nn.Conv2d(100, 2, kernel_size=1, padding=0)

    def forward(self, x):
        #print('first data=', x.data.shape)
        #first conv block
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.pool1(out) 
        #print('out conv1=', out.data.shape)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.pool2(out) 
        #print('size after second pooling', out.shape)
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.pool3(out) 
        #print('size after last pooling', out.shape)
        #print('out after 3rd block',out.shape)
        out = self.fc1(out)
        #print('out after first Fc', out.shape)
        out = self.fc2(out) #Bxnx1xt
        #print('size after all net', out.shape) #bxnx1xt
        out = out.squeeze(dim = 2) #bxnxt
        out = out.transpose(0,1) #nxbxt
        #print('size after squeeze and 
        n, b, t = out.size()
        out = out.contiguous()
        out = out.view(-1, t*b) #nx(t*b)
        out = out.t() #(t*b)xn
        #print('size final output', out.shape)

        return out