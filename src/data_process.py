import sys
import os
import glob
import shutil
import argparse
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

HOME_DIR = os.environ['HOME']
DATA_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'data-schenzen', 'raw')
OUTPUT_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'data-schenzen', 'SKFOLD_10')
create_folder(OUTPUT_DIR)

SEED = 13

input_files = dict()
file_name = 'CHNCXR_*_0.png'
input_files['H0'] = glob.glob(os.path.join(DATA_DIR, file_name))
file_name = 'CHNCXR_*_1.png'
input_files['H1'] = glob.glob(os.path.join(DATA_DIR, file_name))

n_H0 = len(input_files['H0'])
n_H1 = len(input_files['H1'])
print('H0: %i entries' % n_H0)
print('H1: %i entries' % n_H1)

X = np.asarray(input_files['H0'] + input_files['H1'])
y_0 = np.zeros(n_H0)
y_1 = np.ones(n_H1)
y = np.concatenate((y_0, y_1))

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

fold = 0
for trn_idx, val_idx in kfold.split(X, y):
    X_trn, X_val = X[trn_idx], X[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]

    print('Fold #%i' % fold)
    trn_cnt = Counter(y_trn)
    val_cnt = Counter(y_val)
    print(' - Train: %i/%i' % (trn_cnt[0.0], trn_cnt[1.0]))
    print(' - Valid: %i/%i' % (val_cnt[0.0], val_cnt[1.0]))

    # create folders
    H0_trn_dir = os.path.join(OUTPUT_DIR, 'FOLD%i' % fold, 
        'TRAIN', 'NOT_TB')
    H0_val_dir = os.path.join(OUTPUT_DIR, 'FOLD%i' % fold, 
        'VALID', 'NOT_TB')
    create_folder(H0_trn_dir)
    create_folder(H0_val_dir)

    for i in X_trn[np.where(y_trn==0.0)]:
        shutil.copy(i, H0_trn_dir)

    for i in X_val[np.where(y_val==0.0)]:
        shutil.copy(i, H0_val_dir)

    
    H1_trn_dir = os.path.join(OUTPUT_DIR, 'FOLD%i' % fold, 
        'TRAIN', 'TB')
    H1_val_dir = os.path.join(OUTPUT_DIR, 'FOLD%i' % fold, 
        'VALID', 'TB')
    create_folder(H1_trn_dir)
    create_folder(H1_val_dir)

    for i in X_trn[np.where(y_trn==1.0)]:
        shutil.copy(i, H1_trn_dir)

    for i in X_val[np.where(y_val==1.0)]:
        shutil.copy(i, H1_val_dir)    

    fold += 1
    print('----')
print('END')