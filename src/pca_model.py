import sys
import os
import glob
import shutil
import pickle

import tensorflow as tf

import numpy as np

from utils import parse_images, create_folder, load_filenames, check_file
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

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
    # -------------------- #
    #    LOAD FILENAMES    #
    # ==================== #
    X, y = load_filenames(DATA_DIR, dataset_name='schenzen')

    # --------------------------------------- #
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
        
        # --------------------------------- #
        #    TENSORFLOW DATASET PIPELINE    #
        # ================================= #
        width = 128
        height = 128
        channels = 1
        batch_size = 64

        train_files_ds = tf.data.Dataset.from_tensor_slices(X_trn)
        train_ds = train_files_ds.map(
            lambda file: parse_images(file, width, height, channels))
        train_ds = train_ds.batch(batch_size)

        valid_files_ds = tf.data.Dataset.from_tensor_slices(X_val)
        valid_ds = valid_files_ds.map(
            lambda file: parse_images(file, width, height, channels))
        valid_ds = valid_ds.batch(len(y_val))

        # Load TensorFlow original trained model
        file_name = 'cnn_fold%i.h5' % fold
        input_file = os.path.join(CNN_OUTPUTS_DIR, file_name)
        check_file(input_file)
        orig_model = tf.keras.models.load_model(input_file)

        # Remove dense layer and feed forward training dataset
        model = tf.keras.Model(inputs=orig_model.input, 
                               outputs=orig_model.get_layer('dense').output)

        # Feed forward training data
        #embeddings = model.predict(train_ds, verbose=1)
        embeddings = model.predict(valid_ds, verbose=1) # TESTING OF SCRIPT WITH SMALL DATA

        # --------------- #
        #    PCA MODEL    #
        # =============== #
        n_comp = 2
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_new = pca.fit_transform(embeddings)

        variance = pca.explained_variance_ratio_

        print('\n - Auxiliary Info to Build SOM')
        print('  First variance: %1.2f' % variance[0])
        print('  Second variance: %1.2f' % variance[1])
        print('  Ratio = %1.2f\n' % (variance[0]/variance[1]))

        N = 5 * np.sqrt(len(y_trn))
        print('  N = %1.1f\n' % N)



