# Code that evaluates precision and recall for cross-validation experiment.
# It covers all prediction*.npy files and compares them element-wise with their corresponding target*.npy
# The code prints out stats for that experiment.
# Prediction must be discrete at this point.

import os
import numpy as np
import scipy as sp
import scipy.signal

p = []
t = []

list_of_files = os.listdir(os.getcwd()) #list of files in the current directory
for each_file in list_of_files:
    if each_file.startswith('pred'):  #since its all type str you can simply use startswith
        pi = np.load(each_file)
        #print(len(pi))
        p = np.concatenate((p, pi))

    elif each_file.startswith('target'):
    	ti = np.load(each_file)
    	#print(len(ti))
    	t = np.concatenate((t, ti))

m = 121
p_m = sp.signal.medfilt(p, [m])

TN = 0
TP = 0
FN = 0
FP = 0

for i in range(p.size):
	if p[i] == 0 and t[i] == 0:
		TN += 1
	elif p[i] == 0 and t[i] == 1:
		FN += 1
	elif p[i] == 1 and t[i] == 0:
		FP += 1
	elif p[i] == 1 and t[i] == 1:
		TP += 1

precision = TP/(TP + FP)
recall = TP/(TP + FN)

print('TN =', TN, 'TP =', TP, 'FN =', FN, 'FP =', FP, 'precision=', precision, 'recall', recall)
