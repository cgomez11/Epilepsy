#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:08:17 2017

@author: jc.perez13
"""
import numpy as np
import h5py
import scipy.io as sio

base_path = '/Volumes/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/'

#################### train ########################
train =h5py.File(base_path + 'DataW4Aug/train/train_scaled2.mat','r')
t=train.get('train_scaled')

train_data = np.array(t)
train_data = train_data.astype(np.int64)
train_data = np.transpose(train_data)

train_label_dicc = sio.loadmat(base_path +'DataW4Aug/train/train_label2', appendmat = True)
train_label =  train_label_dicc['train_label']


#################### val ########################
val = h5py.File(base_path + 'DataW4Aug/val/val_scaled2.mat','r')
v = val.get('val_scaled')

val_data = np.array(v)
val_data = val_data.astype(np.int64)
val_data = np.transpose(val_data)

val_label_dicc = sio.loadmat(base_path +'DataW4Aug/val/val_label2', appendmat = True)
val_label =  val_label_dicc['val_label']

#################### test ########################
test = h5py.File(base_path + 'DataW4Aug/test/test_scaled2.mat','r')
te = test.get('test_scaled')

test_data = np.array(te)
test_data = test_data.astype(np.int64)
test_data = np.transpose(test_data)

test_label_dicc = sio.loadmat(base_path +'DataW4Aug/test/test_label2', appendmat = True)
test_label =  test_label_dicc['test_label']

np.save(base_path + 'DataW4Aug/dataNN2/train_data.npy', train_data)
np.save(base_path + 'DataW4Aug/dataNN2/train_label.npy', train_label)

np.save(base_path + 'DataW4Aug/dataNN2/val_data.npy', val_data)
np.save(base_path + 'DataW4Aug/dataNN2/val_label.npy', val_label)

np.save(base_path + 'DataW4Aug/dataNN2/test_data.npy', test_data)
np.save(base_path + 'DataW4Aug/dataNN2/test_label.npy', test_label)