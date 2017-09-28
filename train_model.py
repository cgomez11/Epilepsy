# -*- coding: utf-8 -*-

import time
import argparse
import os.path as osp

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

#GPS
import torch.utils.data as data_utils
import numpy as np 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#GPS

#from model_nod3 import Net
import our_network


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--gamma', type=float, default=0.0005, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')
parser.add_argument('--outf', default='/home/gaperezs/disco2/nodules_pyTorch/checkpoints/V177f', help='folder to output images and model checkpoints')
parser.add_argument('--resume', default='', help="path to model (to continue training)") #/home/gaperezs/disco2/nodules_pyTorch/checkpoints/model_epoch_3.pth

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#train_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=True, download=True,
#                   transform=transforms.Compose([
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.1307,), (0.3081,))]
#                   )),
#    batch_size=args.batch_size, shuffle=True, **kwargs)

#GPS
# Load dataset for training and validation (mat2npy)
data_train = np.load('/home/gaperezs/disco2/nodules_pyTorch/dataset/nodules_bd_vT01Hi14M_24i_data_train.npy')
label_train = np.load('/home/gaperezs/disco2/nodules_pyTorch/dataset/nodules_bd_vT01Hi14M_24i_label_train.npy')
#data_val = np.load('/home/gaperezs/disco2/nodules_pyTorch/dataset/nodules_bd_vT01Hi14M_24i_data_val.npy')
#label_val = np.load('/home/gaperezs/disco2/nodules_pyTorch/dataset/nodules_bd_vT01Hi14M_24i_label_val.npy')
meanx = np.mean(data_train, axis=0)
data_train = (data_train - meanx)
np.save('/home/gaperezs/disco2/nodules_pyTorch/dataset/nodules_bd_vT01Hi14M_24i_mean_train_nonorm.npy',meanx)
#np.save('/home/gaperezs/disco2/nodules_pyTorch/dataset/nodules_bd_vT01Hi14M_24i_std_train.npy',stdx)
del meanx
#del stdx
#data_val = data_val - np.mean(data_val, axis=0)
#data_val = np.expand_dims(data_val, axis=1)
data_train = np.expand_dims(data_train, axis=1)
label_train = label_train.astype(np.int64)
#label_val = label_val.astype(np.int64)
#GPS

#GPS
tdata = torch.from_numpy(data_train)
tlabel = torch.from_numpy(np.transpose(label_train))
train = data_utils.TensorDataset(tdata, tlabel)
train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)
#GPS

#test_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=False,
#                   transform=transforms.Compose([
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.1307,), (0.3081,))]
#                   )),
#    batch_size=args.batch_size, shuffle=True, **kwargs)

#GPS
#vdata = torch.from_numpy(data_val)
#vlabel = torch.from_numpy(np.transpose(label_val))
#test = data_utils.TensorDataset(vdata, vlabel)
#test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=True)
#GPS

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.00, 0.01)
        #m.bias.data.normal_(0.00, 0.1)
        m.bias.data.fill_(0.1)
        #xavier(m.weight.data)
        #xavier(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        #m.weight.data.normal_(1.0, 0.01)
        #m.bias.data.fill_(0)


model = Net()
#GPS
model.apply(weights_init)
res_flag = 0
if args.resume != '': # For training from a previously saved state
    model.load_state_dict(torch.load(args.resume))
    res_flag = 1
print(model)
#GPS
 
if args.cuda:
    model.cuda()

load_model = False
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        load_model = True

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.CrossEntropyLoss() # nn.BCELoss().cuda() #nn.SoftMarginLoss()


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.squeeze(target))
        train_loss += loss.data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        acccuracy_batch = 100. * correct / (len(data)*(batch_idx+1.0))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} ({:.3f})\tAcc: {:.2f}% '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], train_loss/(batch_idx+1.0), acccuracy_batch))
            #print(output)

"""
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, torch.squeeze(target)).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        acccuracy_batch = 100. * correct / (len(data)*(batch_idx+1.0))
        if batch_idx % args.log_interval == 0:
            print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(test_loader.dataset),
                100. * batch_idx / len(test_loader), test_loss/(batch_idx+1.0), acccuracy_batch))
    test_loss = test_loss
    # loss function already averages over batch size
    test_loss /= len(test_loader)
    acccuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,
                                               correct,
                                               len(test_loader.dataset),
                                               acccuracy))
    return test_loss
"""

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed
       by 10 at every specified step
       Adapted from PyTorch Imagenet example:
       https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    best_loss = None
    if load_model:
        best_loss = test(0)
    #GPS
    if res_flag == 0:
	Ei = 1
    else:
	if args.resume[-6] == '_':
	    Ei = int(args.resume[-5]) + 1
            print('-' * 89)
            print('Resuming from epoch %d' % (Ei))
            print('-' * 89)
        else:
	    Ei = int(args.resume[-6:-4]) + 1
            print('-' * 89)
            print('Resuming from epoch %d' % (Ei))
            print('-' * 89)
    #GPS
    try:
        for epoch in range(Ei, args.epochs + 1):
            epoch_start_time = time.time()
            train(epoch)
            #test_loss = test(epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s ({:.2f}h)'.format(
                epoch, time.time() - epoch_start_time, (time.time() - epoch_start_time)/3600.0))
            print('-' * 89)
            #if best_loss is None or test_loss < best_loss:
                #best_loss = test_loss
                #with open(args.save, 'wb') as fp:
                    #state = model.state_dict()
                    #torch.save(state, fp)
            torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (args.outf, epoch)) #GPS
            #else:
                #adjust_learning_rate(optimizer, args.gamma, epoch)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


