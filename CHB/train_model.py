# -*- coding: utf-8 -*-
import os
import h5py
import time
import copy
import tqdm
import argparse
import numpy as np 
import scipy.io as sio
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau


from  model.network_FC1D import Net
# from model.network_FC1D_50 import Net

parser = argparse.ArgumentParser(description='PyTorch Epilepsy')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--gamma', type=float, default=0.0005, metavar='M',
                    help='learning rate decay factor (default: 0.0005)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--gpuNum', type=str, default='3',
                    help='number of gpu for test')
parser.add_argument('--foldNum', type=str, default='Exp1',
                    help='fold to evaluate in 3 fold cross val')
parser.add_argument('--config', type=str, default='FIRST',
                    help='configuration to load data')
parser.add_argument('--data_path', type=str, default='media/user_home2/EEG/Epilepsy/data',
                    help='folder that contains the database for training')
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')

parser.add_argument('--outf', default='/media/user_home2/EEG/Epilepsy/models', help='folder to output model')
parser.add_argument('--resume', default='', help="path to model (to continue training)")
parser.add_argument('--finetune', default='', help="path to model weights for finetuning")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# set GPU number
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuNum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset for training and validation
path_to_data = args.data_path
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, transform=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        y = self.labels[ID]
        X = np.load(path_to_data +ID + '.npy')
        X = np.transpose(X) #23x1024
        X = np.expand_dims(X, axis=0) #1x23x1024
        if self.transform:
            X = self.transform(X)
        return X, y


path_to_dicts = '/media/user_home1/EEG/Epilepsy/data/configs/'

partition = np.load(path_to_dicts + 'partitions_' + args.config + args.foldNum+'.npy').item()
labels = np.load(path_to_dicts + 'labels_' + args.config + args.foldNum + '.npy').item()


image_datasets = {x: Dataset(partition[x], labels,transform=transforms.Compose([transforms.ToTensor()])) for x in ['train', 'validation']} 
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'validation']}
train_loader = dataloaders['train']
test_loader = dataloaders['validation']
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
print('training dataset size:', dataset_sizes['train'])
print('Validation dataset size:', dataset_sizes['validation'])
print('Done creating dataloader \n')

# custom weights initialization called on netG and netD
## 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.00, 0.01)
        m.bias.data.fill_(0.1)
    elif classname.find('GroupNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()

# Load Network
model = Net()

model.apply(weights_init)
res_flag = 0
if args.finetune != '': # For training from a previously saved state
    model.load_state_dict(torch.load(args.finetune))
    res_flag = 1
    # freeze and unfreeze layers
    for param in model.parameters():
        param.requires_grad = False
        
print(model)
 
if args.cuda:
    model.cuda()

load_model = False
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        load_model = True

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.CrossEntropyLoss() 


def train(epoch, criterion, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    running_loss = 0.0
    running_corrects = 0
    for  data, target in tqdm.tqdm(dataloaders['train'], total=len(dataloaders['train']), desc='Batch'):

        data = data.to(device)
        data = torch.squeeze(data)
        data = data.float()
        target = target.to(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.data
        sm = nn.Softmax(dim=1)
        output_sm = sm(output)
        _, preds = torch.max(output_sm, 1)
        running_loss += loss.item() * data.size(0)
        running_corrects += torch.sum(preds == target.data)
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / dataset_sizes['train']

    epoch_acc = running_corrects.double() / dataset_sizes['train']
    line_to_save_train = 'Train set: Average loss: {:.4f} Accuracy: {}/{} {:.4f}\n'.format(epoch_loss,
                                               running_corrects,
                                               len(train_loader.dataset),
                                               epoch_acc)
    with open(args.outf+'/ACC_train.txt','a') as f:
        f.write(line_to_save_train)
    print(line_to_save_train)


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    running_loss = 0.0
    running_corrects = 0
    for  data, target in tqdm.tqdm(dataloaders['validation'], total=len(dataloaders['validation']), desc='Batch'):
        data = data.to(device)
        data = data.float()
        data = torch.squeeze(data)
        target = target.to(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        sm = nn.Softmax(dim=1)
        output_sm = sm(output)
        _, preds = torch.max(output_sm, 1)
        loss = criterion(output, target)
        running_loss += loss.item() * data.size(0)
        running_corrects += torch.sum(preds == target.data)
    epoch_loss = running_loss / dataset_sizes['validation']

    epoch_acc = running_corrects.double() / dataset_sizes['validation']
    line_to_save_test = 'Test set: Average loss: {:.4f} Accuracy: {}/{} {:.4f}\n'.format(epoch_loss,
                                               running_corrects,
                                               dataset_sizes['validation'],
                                               epoch_acc)

    with open(args.outf+'/ACC_test.txt','a') as f:
        f.write(line_to_save_test)
    print(line_to_save_test)

    return epoch_loss, epoch_acc


def adjust_learning_rate(optimizer, epoch):  
    #lr = args.lr * (gamma ** (step))
    lr = args.lr * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 90))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    best_loss = None
    if load_model:
        best_loss = test(0)
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
    try:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(Ei, args.epochs + 1):
            epoch_start_time = time.time()
            train(epoch, criterion, optimizer)
            test_loss, test_acc = test(epoch)
            if test_acc> best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                filename = args.outf + '/complete_model'+ '_BEST.pth'
                state = {'epoch': epoch , 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict() }
                torch.save(state, filename)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s ({:.2f}h)'.format(
                epoch, time.time() - epoch_start_time, (time.time() - epoch_start_time)/3600.0))
            print('-' * 89)
            # Save trained model
            filename = args.outf + '/complete_model_' + str(epoch)+ '.pth'
            state = {'epoch': epoch , 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict() }
            torch.save(state, filename)
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
