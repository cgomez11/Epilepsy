#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np

#read .mat files
train_scaled_dicc = sio.loadmat('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/train_scaled', appendmat = True)
train_data = train_scaled_dicc['train_scaled']

train_data = train_data.astype(np.int64)

train_label_dicc = sio.loadmat('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/train_label', appendmat = True)
train_label =  train_label_dicc['train_label']


val_scaled_dicc = sio.loadmat('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/val_scaled', appendmat = True)
val_data = val_scaled_dicc['val_scaled']

val_data = val_data.astype(np.int64)

val_label_dicc = sio.loadmat('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/val_label', appendmat = True)
val_label =  val_label_dicc['val_label']


test_scaled_dicc = sio.loadmat('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/test_scaled', appendmat = True)
test_data = test_scaled_dicc['test_scaled']

test_data = test_data.astype(np.int64)

test_label_dicc = sio.loadmat('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/test_label', appendmat = True)
test_label =  test_label_dicc['test_label']

#save as .npy files
np.save('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/train_data.npy', train_data)
np.save('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/train_label.npy', train_label)

np.save('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/val_data.npy', val_data)
np.save('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/val_label.npy', val_label)

np.save('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/test_data.npy', test_data)
np.save('/media/oem/ADATA_HV100/Cata/Fronteras/Project/Dataset/DenseData/DataW4/DataExp1/test_label.npy', test_label)