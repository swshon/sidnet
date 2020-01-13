import os,sys
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './models')
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy.special import softmax


def calculate_eer(scores,test_trials):
        test_trials = map(int,test_trials)
        fpr,tpr,threshold = roc_curve(test_trials,scores,pos_label=1)
        fnr = 1-tpr
        EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
        EER = fpr[np.argmin(np.absolute((fnr-fpr)))]
        return EER

    