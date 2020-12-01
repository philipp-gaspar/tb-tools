import sys
import os
import glob
import shutil
import pickle

import tensorflow as tf

import numpy as np

from utils import parse_images, create_folder, load_filenames
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# Define global variables
HOME_DIR = os.environ['HOME']
DATA_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'data-schenzen', 'raw')
PACKAGE_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'tb-tools')

EXPERIMENTS_DIR = os.path.join(PACKAGE_DIR, 'experiments')
OUTPUTS_DIR = os.path.join(EXPERIMENTS_DIR, 'PCA', 'outputs')
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'PCA', 'results')
create_folder(OUTPUTS_DIR)
create_folder(RESULTS_DIR)
CNN_OUTPUTS_DIR = os.path.join(EXPERIMENTS_DIR, 'CNN', 'outputs')

SEED = 13

if __name__ == "__main__":
    # ==================== #
    #    LOAD FILENAMES    #
    # ==================== #
    X, y = load_filenames(DATA_DIR, dataset_name='schenzen')

    # ======================================= #
    #    STRATIFIED KFOLD CROSS VALIDATION    #
    # ======================================= #
    kfold = StratifiedKFold(n_splits=10, 
                            shuffle=True, 
                            random_state=SEED)

    fold = 0
    for trn_idx, val_idx in kfold.split(X, y):
        fold += 1
        print('Fold #%i' % fold)

        # Shuffle indexes
        np.random.shuffle(trn_idx)
        np.random.shuffle(val_idx)

        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]

        # Count entries for each class
        trn_cnt = Counter(y_trn)
        val_cnt = Counter(y_val)
        print(' - Train: %i/%i' % (trn_cnt[0.0], trn_cnt[1.0]))
        print(' - Valid: %i/%i' % (val_cnt[0.0], val_cnt[1.0]))

        # Load TensorFlow trained model
        file_name = 'cnn_fold%i.h5' % fold
        input_file = os.path.join(CNN_OUTPUTS_DIR, file_name)
        print(input_file)
        
        
        width = 128
        height = 128
        channels = 1



