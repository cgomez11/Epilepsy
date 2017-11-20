# Code that evaluates precision and recall for cross-validation experiment.
# It covers all prediction*.npy files and compares them element-wise with their corresponding target*.npy
# The code prints out stats for that experiment.
# Prediction must be discrete at this point. 
###### Changes mst be made to accomodate iteration over confidence

import os
import numpy as np
import scipy as sp
import scipy.signal

# Constants
pat_num = 21

# Gets record names to open preds and targets simultaneosuly
R = []
with open("stats.txt") as st:

	for num, line in enumerate(st):
		
		RecordN, Record, SpecificityN, Specificity, RecallN, Recall, PrecisionN, Precision = line.split (' ')
		R.append(Record)
    
# Opens preds and target files in order. Loads into arrays.
# Concatenates into single one-dimensional array           
p = []
t = []

for i in range(len(R)):
	filename_pred = 'pred_' + R[i] + '_21.npy'
	filename_target = 'target_' + R[i] + '_21.npy'

	pi = np.load(filename_pred)
	p = np.concatenate((p, pi))

	ti = np.load(filename_target)
	t = np.concatenate((t, ti))

# Makes sure both preds and targets have the same size
print(p.size)
print(t.size)

# Counts True Positives and Negatives; and False Positives and Negatives
TN = 0
TP = 0
FN = 0
FP = 0

for i in range(len(p)-1):
	if p[i] == 0 and t[i] == 0:
		TN += 1
	elif p[i] == 0 and t[i] == 1:
		FN += 1
	elif p[i] == 1 and t[i] == 0:
		FP += 1
	elif p[i] == 1 and t[i] == 1:
		TP += 1

# Computes Precision and Recall
precision = TP/(TP + FP)
recall = TP/(TP + FN)

print('TN =', TN, 'TP =', TP, 'FN =', FN, 'FP =', FP, 'precision=', precision, 'recall', recall)

#########
### Filtering to supress false positives
##### Same as before but for filtered signal
#########
m = 61
p_m = sp.signal.medfilt(p, [m])

TN = 0
TP = 0
FN = 0
FP = 0

for i in range(p_m.size):
	if p_m[i] == 0 and t[i] == 0:
		TN += 1
	elif p_m[i] == 0 and t[i] == 1:
		FN += 1
	elif p_m[i] == 1 and t[i] == 0:
		FP += 1
	elif p_m[i] == 1 and t[i] == 1:
		TP += 1


precision = TP/(TP + FP)
recall = TP/(TP + FN)
print('median filter=', m)
print('TN =', TN, 'TP =', TP, 'FN =', FN, 'FP =', FP, 'precision=', precision, 'recall', recall)
