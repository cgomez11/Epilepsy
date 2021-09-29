# Code that calculates PR curve for each patient with different threshold values for the positive class
import time
import argparse
import glob, os
import numpy as np
import scipy as sp
import scipy.signal
import scipy.io as sio

from pandas_ml import ConfusionMatrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score

############# input parameters 

parser = argparse.ArgumentParser(description='PyTorch Epilepsy PR code')
parser.add_argument('--exp_path', default='/media/user_home2/EEG/Epilepsy/results', 
                    help='experiment folder where model predictions were saved')
parser.add_argument('--save_path', type=str, default='/media/user_home2/EEG/Epilepsy/results',
                    help='path to save PR curve and stats')
parser.add_argument('--validate', type=str, default='test',
                    help='validate over test or train set')

args = parser.parse_args()

assert(os.path.exists(args.exp_path))
assert(os.path.exists(args.save_path))


############## LOO 
#pat_num = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
#pat_num = ['08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']

############## Main code

folds = ['Exp1','Exp2','Exp3']

for foldNum in folds:
    Fmax = []
    F_avg = []
    F_std = []
    AP_all = []
    AP_avg = []
    AP_std = []
    pat_list = []
    best_precision = []
    best_recall = []
    BP_avg = []
    BP_std = []
    BR_avg = []
    BR_std = []
    ACC = []
    ACC_avg = []
    ACC_std = []

    SPE = []
    SPE_avg = []
    SPE_std = []

    if args.validate == 'test':  
        ############## 3FCV test patient division
        if foldNum == 'Exp1':
            test_patients = ['16','14','19','17','13','01','24','15']
        elif foldNum == 'Exp2':
            test_patients = ['02','22','20','18','03','10','11','12']
        elif foldNum == 'Exp3':
            test_patients = ['06','21','09','07','04','23','05','08']
    elif args.validate == 'train':
        if foldNum == 'Exp1':
            test_patients = ['02','22','20','18','03','10','11','12', '06','21','09','07','04','23','05','08']
        elif foldNum == 'Exp2':
            test_patients = ['16','14','19','17','13','01','24','15', '06','21','09','07','04','23','05','08']
        elif foldNum == 'Exp3':
            test_patients = ['16','14','19','17','13','01','24','15', '02','22','20','18','03','10','11','12' ]


    base_path = args.exp_path + foldNum + '/'
    dire = os.listdir(base_path)
    for f in range(len(test_patients)):

        pat_list.append(int(test_patients[f]))
        output_per_patient = [x for x in dire if x.startswith(('output_chb%s')%(test_patients[f]))]

        full_out = [] # to concatenate all the outputs of patient f
        full_target = [] # to concatenate all the targets of patient f

    #concatenate all outputs and targets
        for j in range(len(output_per_patient)):
            out = np.load(base_path + output_per_patient[j])
            out  = out[:,1]
            target = np.load(base_path + 'target' + output_per_patient[j][6:])
            full_out = np.concatenate((full_out, out), axis=0)
            full_target = np.concatenate((full_target, target), axis=0)

        precision = [] 
        recall = []

        # Compute precision-recall pairs for different probability thresholds
        precision, recall, thresholds = precision_recall_curve(full_target, full_out, pos_label=1)
        AP = average_precision_score(full_target, full_out) 
        AP_all.append(AP)
        #Calculate F-measure
        P_arr = np.array(precision)  
        R_arr = np.array(recall) 
        F =  2*(P_arr*R_arr)/(P_arr + R_arr+1e-06)
        f_score = np.amax(F)
        Fmax.append(f_score)
        pos_f_score = np.argmax(F)
        # find the threshold that corresponds to the best f_score
        best_thresh = thresholds[pos_f_score]
        #precision and recall of the best threshold 
        best_precision.append(P_arr[pos_f_score])
        best_recall.append(R_arr[pos_f_score])
        binary_labels = np.array(full_out) 
        binary_labels[binary_labels > best_thresh] = 1
        binary_labels[binary_labels < best_thresh] = 0
        CM = ConfusionMatrix(full_target, binary_labels)
        SPE.append(CM.TNR)
        ACC.append(CM.ACC)
        file_name_labels = args.save_path + '/' + foldNum + '_pred_' + test_patients[f] + '.npy'
        np.save(file_name_labels, binary_labels)
        #save complete PR curve
        results = np.stack((precision, recall), axis=1) 
        filename = args.save_path + '/'+foldNum + '_PR_' + test_patients[f] + '.npy'
        np.save(filename, results)
        line_to_print = 'Done patient {}'.format(test_patients[f])
        print(line_to_print)

    #save txt of each fold 
    results_f_patient = np.stack((pat_list, Fmax, AP_all, best_precision, best_recall, ACC, SPE), axis=1)
    filename_f_patient = args.save_path + '/'+foldNum  +'_fMeasure_AP_per_patient.txt'
    np.savetxt(filename_f_patient, results_f_patient, fmt=['%u', '%.4e', '%.4e', '%.4e', '%.4e', '%.4e', '%.4e'], header = 'Patient Fmeasu AverageP Precision Recall Accura Speci')


    F_avg = np.mean(np.array(Fmax))
    F_std = np.std(np.array(Fmax))
    AP_avg = np.mean(np.array(AP_all))
    AP_std = np.std(np.array(AP_all))
    BP_avg = np.mean(np.array(best_precision))
    BP_std =np.std(np.array(best_precision))
    BR_avg = np.mean(np.array(best_recall))
    BR_std = np.std(np.array(best_recall))
    ACC_avg = np.mean(np.array(ACC))
    ACC_std = np.std(np.array(ACC))
    SPE_avg = np.mean(np.array(SPE))
    SPE_std = np.std(np.array(SPE))

    filename_f = args.save_path + '/'+foldNum+'_fMax_global.txt'
    results_f = np.column_stack((F_avg, F_std, AP_avg, AP_std, BP_avg, BP_std, BR_avg, BR_std, ACC_avg, ACC_std, SPE_avg, SPE_std))
    np.savetxt(filename_f, results_f, fmt='%.4e', header = 'Fmax_avg  Fmax_std  AvPre_avg  AvPre_std  Preci_avg  Preci_std  Reca_avg  Reca_std ACC_avg ACC_std SPE_avg SPE_std')
