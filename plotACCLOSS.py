#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:08:56 2017

@author: oem
"""

import numpy as np
import matplotlib.pyplot as plt

base_path = '/media/oem/CATALINA/Epilepsia/Networks/DenseData/DataW41sAdj/bigNet'

########################## train ###########################
train1 = np.genfromtxt(base_path + '/Exp1/ACC_train.txt')
loss1 = train1[:,4]
acc1 = train1[:,7]
epoch = np.linspace(1, 150, num=150)

test1 = np.genfromtxt(base_path + '/Exp1/ACC_test.txt')
loss_te1 = test1[:,4]
acc_te1 = test1[:,7]


train2 = np.genfromtxt(base_path + '/Exp2/ACC_train.txt')
loss2 = train2[:,4]
acc2 = train2[:,7]

test2 = np.genfromtxt(base_path + '/Exp2/ACC_test.txt')
loss_te2 = test2[:,4]
acc_te2 = test2[:,7]

train3 = np.genfromtxt(base_path + '/Exp3/ACC_train.txt')
loss3 = train3[:,4]
acc3 = train3[:,7]

test3 = np.genfromtxt(base_path + '/Exp3/ACC_test.txt')
loss_te3 = test3[:,4]
acc_te3 = test3[:,7]


fig1=plt.figure()
plt.plot(epoch, acc1, 'c', label='ACC1')
plt.plot(epoch, acc2, 'm', label='ACC2')
plt.plot(epoch, acc3, 'b', label='ACC3')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Train ACC: 3 block NN')
plt.legend()


fig2=plt.figure()
plt.plot(epoch, loss1, 'c', label = 'Loss1')
plt.plot(epoch, loss2, 'm', label = 'Loss2')
plt.plot(epoch, loss3, 'b', label = 'Loss3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train loss: 3 block NN')
plt.legend()

########################## test ###########################
fig3=plt.figure()
plt.plot(epoch, acc_te1, 'c', label='ACC1')
plt.plot(epoch, acc_te2, 'm', label='ACC2')
plt.plot(epoch, acc_te3, 'b', label='ACC3')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test ACC: 3 block NN')
plt.legend()


fig4=plt.figure()
plt.plot(epoch, loss_te1, 'c', label = 'Loss1')
plt.plot(epoch, loss_te2, 'm', label = 'Loss2')
plt.plot(epoch, loss_te3, 'b', label = 'Loss3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test loss: 3 block NN')
plt.legend()

##### average values:train #######
acc1=np.expand_dims(acc1, axis=1)
acc2= np.expand_dims(acc2, axis=1)
acc3 =np.expand_dims(acc3, axis=1)
all_acc_train = np.concatenate((acc1, acc2, acc3), axis= 1)

avg_acc_train=all_acc_train.mean(axis=1)
std_acc_train=all_acc_train.std(axis=1)

fig5=plt.figure()
plt.errorbar(epoch, avg_acc_train, std_acc_train, linestyle='None', marker='.',ecolor='gray',lw=1)
plt.title('Average ACC train: 3 block NN')
plt.xlabel('Epoch')
plt.ylabel('Average Training Accuracy')

##### average values:test #######
acc_te1=np.expand_dims(acc_te1, axis=1)
acc_te2= np.expand_dims(acc_te2, axis=1)
acc_te3 =np.expand_dims(acc_te3, axis=1)

all_acc_test = np.concatenate((acc_te1, acc_te2, acc_te3), axis= 1)
avg_acc_test=all_acc_test.mean(axis=1)
std_acc_test=all_acc_test.std(axis=1)  
fig6=plt.figure()
plt.errorbar(epoch, avg_acc_test, std_acc_test, linestyle='None', marker='.',ecolor='gray',lw=1)
plt.title('Average ACC test: 3 block NN')
plt.xlabel('Epoch')
plt.ylabel('Average Test Accuracy')


