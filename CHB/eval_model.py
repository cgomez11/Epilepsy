#! /usr/bin/env python
import os
import glob
import h5py
import time
import argparse
import numpy as np 
import os.path as osp
import scipy.io as sio

from pandas_ml import ConfusionMatrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data_utils
from torchvision import datasets, transforms


from network_FC1D  import Net


############## input arguments

parser = argparse.ArgumentParser(description='PyTorch Epilepsy')

parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--validate', type=str, default='test',
                   help='validate over test or train set')
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')
parser.add_argument('--gpuNum', type=str, default='3',
                    help='number of gpu for test')
parser.add_argument('--foldNum', type=str, default='Exp1',
                    help='fold to evaluate in 3 fold cross val')
parser.add_argument('--epoch', type=str, default='0',
                    help='epoch of trained model for test')
parser.add_argument('--outlier_type', type=str, default='thresh',
                    help='method of removing outliers (thresh or stats)')
parser.add_argument('--path', default='/media/user_home2/EEG/Epilepsy/results', 
                    help='folder to output images and model checkpoints')
parser.add_argument('--dataPath',type =str, default='/media/user_home2/EEG/Epilepsy/data/',
                    help='folder to load stats')
parser.add_argument('--model_path', default='/home/cgomez11/Project/Epilepsy/Checkpoints/DenseData/DataW4_025s_adj/prueba9/Exp3/model_epoch_32.pth', 
                    help="path to model to evaluate") 

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

assert(os.path.exists(args.path))
assert(os.path.exists(args.model_path))

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# set GPU number
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuNum

model = Net()

if args.model_path != '': # For training from a previously saved state
    #model.load_state_dict(torch.load(args.model_path))
    checkpoint = torch.load(args.model_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
print(model)
#GPS
 
if args.cuda:
    model.cuda()

"""
load_model = False
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        load_model = True
"""

def test_eval(ite, epoch, pat_record, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.float()#.squeeze()
        data = torch.squeeze(data,1)
        data = torch.transpose (data, 1, 2) #bxcx1024
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        sm = nn.Softmax(dim=1)
        output_sm = sm(output)
        
        target = target.contiguous()
        target = target.view(target.data.shape[0]*target.data.shape[1])

        pred = output_sm.data.abs()
        output_to_save = pred.cpu().numpy()
        np.save(args.path + 'output_'+ pat_record + str(ite)+'_'+ str(epoch) + '.npy', output_to_save)

        # Save labels in same format as predictions
        target_to_save = target.data.cpu().numpy()
        target_to_save = target_to_save.astype(np.int64)
        np.save(args.path + 'target_'+ pat_record + str(ite) + '_' + str(epoch)+ '.npy', target_to_save)

    line_to_save_test = 'Done {}'.format(pat_record)
    #with open(path + 'stats.txt','a') as f:
    #    f.write(line_to_save_test)
    print(line_to_save_test)


# define set for 2 fold cross val
if args.validate == 'test':
    if args.foldNum == 'Exp1':
        test_patients = ['16','14','19','17','13','01','24','15']
    elif args.foldNum == 'Exp2':
        test_patients = ['02','22','20','18','03','10','11','12']
    elif args.foldNum == 'Exp3':
        test_patients = ['06','21','09','07','04','23','05','08']
elif args.validate == 'train':
    if args.foldNum == 'Exp1':
        test_patients = ['02','22','20','18','03','10','11','12', '06','21','09','07','04','23','05','08']
    elif args.foldNum == 'Exp2':
        test_patients = ['16','14','19','17','13','01','24','15', '06','21','09','07','04','23','05','08']
    elif args.foldNum == 'Exp3':
        test_patients = ['16','14','19','17','13','01','24','15', '02','22','20','18','03','10','11','12' ]

# define method for saturating outliers
if args.outlier_type == 'thresh':
    thresh_upper = 250 
    thresh_lower = -250
elif args.outlier_type == 'stats':
    complete_data_pat = args.dataPath
    print(complete_data_pat)
    mu = sio.loadmat(complete_data_pat + 'media_train_3FCV.mat')
    mu = mu['media']
    mu = mu[0,0]
    sigma = sio.loadmat(complete_data_pat + 'sigma_train_3FCV.mat')
    sigma = sigma['sigma']
    sigma = sigma[0,0]
    thresh_upper = mu + 3*sigma
    thresh_lower = mu - 3*sigma

for id in range(len(test_patients)):

    files_records = glob.glob('/media/user_home1/EEG/Epilepsy/patients/chb'+ test_patients[id] +'/*.mat') #evaluate all the records of pacient X
    main_path = '/media/user_home1/EEG/Epilepsy/patients/labels/chb'+ test_patients[id] +'_labels/'
    tot_time = time.time()
    # iterate over all files of each patient
    for f in range(len(files_records)): 
        #read .mat files 
        aux = files_records[f]
        pos = aux.rfind('/')
        pat_record = aux[pos+1:-4]
        print(pat_record)
        record_dic = sio.loadmat(files_records[f])
        record  = record_dic['m_SignalsMat']
        row, cols = np.shape(record)
        one_hour =921600
        proportion = one_hour // 64

        # saturate at thresh upper and thresh lower
        record[record>thresh_upper] = thresh_upper
        record[record<thresh_lower] = thresh_lower

        record = np.expand_dims(record, 2) #23xlongx1
        print('reading...', main_path + pat_record + '_labels.mat')
        label = h5py.File(main_path + pat_record + '_labels.mat')
        label = label.get('labels_pat')
        label = np.array(label)  #1xnumlabels
        print('Done loading data!')

        # if a recording lasts more than 1 hour: split to run inference
        if cols > one_hour: 
            n_hours = cols // one_hour
            for it in range(n_hours):
                print('iteration {}'.format(str(it)))
                temp_record = record[:,it*one_hour:(it+1)*one_hour-1]
                temp_label = label[:,it*proportion:(it+1)*proportion-1]
 
                #adapt data 
                data_test = np.transpose(temp_record)
                data_test = np.expand_dims(data_test, axis=1)
                label_test = temp_label.astype(np.int64)
                label_test = label_test[:,7:-8]

                #dataloader
                testdata = torch.from_numpy(data_test)
                testlabel = torch.from_numpy(label_test)
                test = data_utils.TensorDataset(testdata, testlabel)
                test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=True)
                test_eval(str(it), epoch, pat_record, test_loader)

        else:
            data_test = np.transpose(record)
            data_test = np.expand_dims(data_test, axis=1)
            #print('data test shape', np.shape(data_test))
            label_test = label.astype(np.int64)
            label_test = label_test[:,7:-8]

            #dataloader
            testdata = torch.from_numpy(data_test)
            testlabel = torch.from_numpy(label_test)
            test = data_utils.TensorDataset(testdata, testlabel)
            test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=True)
            test_eval('', epoch, pat_record, test_loader)
    

