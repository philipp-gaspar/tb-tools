import sys
import os
import glob
import shutil
import argparse
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, \
    Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

import numpy as np

from utils import parse_images, EarlyStoppingAtSP, create_folder, load_filenames
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc

HOME_DIR = os.environ['HOME']
DATA_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'data-schenzen', 'raw')
PACKAGE_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'tb-tools')

EXPERIMENTS_DIR = os.path.join(PACKAGE_DIR, 'experiments')
OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, 'CNN', 'outputs')
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'CNN', 'results')
create_folder(OUTPUT_DIR)
create_folder(RESULTS_DIR)

BATCH_SIZE = 64
EPOCHS = 10
SEED = 13
np.random.seed(SEED)

def create_cnn(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, 
                     kernel_size=(3,3), 
                     activation='relu', 
                     input_shape=input_shape)) 
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, 
                     kernel_size=(3,3), 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model
    
if __name__ == "__main__":
    # ==================== #
    #    LOAD FILENAMES    #
    # ==================== #
    X, y = load_filenames(DATA_DIR, dataset_name='schenzen')

    # --- METRIC CONTAINERS --- #
    train_metrics = {'acc': [], 'auc': [], 'tnr': [], 'fnr': [],
                     'tpr': [], 'fpr': [], 'sp': [], 'loss': []}
    valid_metrics = {'acc': [], 'auc': [], 'tnr': [], 'fnr': [],
                     'tpr': [], 'fpr': [], 'sp': [], 'loss': []}

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

        # shuffle indexes
        np.random.shuffle(trn_idx)
        np.random.shuffle(val_idx)

        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]

        # count entries for each class
        trn_cnt = Counter(y_trn)
        val_cnt = Counter(y_val)
        print(' - Train: %i/%i' % (trn_cnt[0.0], trn_cnt[1.0]))
        print(' - Valid: %i/%i\n' % (val_cnt[0.0], val_cnt[1.0]))

        # ================================= #
        #    TENSORFLOW DATASET PIPELINE    #
        # ================================= #
        width = 128
        height = 128
        channels = 1
        
        train_files_ds = tf.data.Dataset.from_tensor_slices(X_trn)
        train_ds = train_files_ds.map(
            lambda file: parse_images(file, width, height, channels))
        train_ds = train_ds.batch(BATCH_SIZE)

        valid_files_ds = tf.data.Dataset.from_tensor_slices(X_val)
        valid_ds = valid_files_ds.map(
            lambda file: parse_images(file, width, height, channels))
        valid_ds = valid_ds.batch(len(y_val))

        # ================= #
        #    TRAIN MODEL    #
        # ================= #
        K.clear_session()
        input_shape = (width, height, channels)
        patience = 3

        model = create_cnn(input_shape)
        early_stop = EarlyStoppingAtSP(
            validation_data=[valid_ds, y_val], 
            patience=3)
        history = model.fit(train_ds, 
                            epochs=1, 
                            validation_data=valid_ds,
                            callbacks=[early_stop], 
                            verbose=1)
        
        # save best epoch
        history.history['best_epoch'] = early_stop.best_epoch
        file_path = os.path.join(RESULTS_DIR, 'history_fold%i.pkl' % fold)
        with open(file_path, 'wb') as fp:
            pickle.dump(history.history, fp)

        # --- LOSS & ACCURACY --- #
        trn_loss, trn_acc = model.evaluate(train_ds)
        train_metrics['acc'].append(trn_acc)
        train_metrics['loss'].append(trn_loss)

        val_loss, val_acc = model.evaluate(valid_ds)
        valid_metrics['acc'].append(val_acc)
        valid_metrics['loss'].append(val_loss)

        # --- ROC CURVE & SP INDEX (TRAIN) --- #
        #y_prob_trn = model.predict(X_trn.reshape(-1,8,8,1))
        y_prob_trn = model.predict(train_ds)
        fpr, tpr, threshold = roc_curve(y_trn, y_prob_trn)
        trn_auc = auc(fpr, tpr)
        train_metrics['auc'].append(trn_auc)

        trn_sp = np.sqrt(np.sqrt(tpr*(1-fpr)) * (0.5*(tpr+(1-fpr))))
        trn_knee = np.argmax(trn_sp)
        train_metrics['sp'].append(trn_sp[trn_knee])

        y_pred_trn = np.zeros((len(y_prob_trn), 1))
        y_pred_trn[y_prob_trn > threshold[trn_knee]] = 1.0
        conf_matrix = confusion_matrix(y_trn, y_pred_trn)
        tn, fp, fn, tp = conf_matrix.ravel()
        train_metrics['tpr'].append(tp/(tp+fn))
        train_metrics['tnr'].append(tn/(tn+fp))
        train_metrics['fpr'].append(fp/(fp+tn))
        train_metrics['fnr'].append(fn/(tp+fn))

        # --- ROC CURVE & SP INDEX (VALID) --- #
        #y_prob_val = model.predict(X_val.reshape(-1,8,8,1))
        y_prob_val = model.predict(valid_ds)
        fpr, tpr, threshold = roc_curve(y_val, y_prob_val)
        val_auc = auc(fpr, tpr)
        valid_metrics['auc'].append(val_auc)

        val_sp = np.sqrt(np.sqrt(tpr*(1-fpr)) * (0.5*(tpr+(1-fpr))))
        val_knee = np.argmax(val_sp)
        valid_metrics['sp'].append(val_sp[val_knee])

        y_pred_val = np.zeros((len(y_prob_val), 1))
        y_pred_val[y_prob_val > threshold[val_knee]] = 1.0
        conf_matrix = confusion_matrix(y_val, y_pred_val)
        tn, fp, fn, tp = conf_matrix.ravel()
        valid_metrics['tpr'].append(tp/(tp+fn))
        valid_metrics['tnr'].append(tn/(tn+fp))
        valid_metrics['fpr'].append(fp/(fp+tn))
        valid_metrics['fnr'].append(fn/(tp+fn))

        # ------------------------------ #
        #    SAVING THE TRAINED MODEL    #
        # ============================== #
        output_name = 'cnn_fold%i.h5' % fold
        model.save(os.path.join(OUTPUT_DIR, output_name), save_format='h5')

    print('END TRAINING\n')

    # ------------------------- #
    #    PERFORMANCE METRICS    #
    # ========================= #
    print('PERFORMANCE METRICS:')
    for key in train_metrics.keys():
        mean_value = np.mean(train_metrics[key])
        std_value = np.std(train_metrics[key])
        print(' - Train [%s]: %1.4f +- %1.4f' %
            (key.upper(), mean_value, std_value))
        mean_value = np.mean(valid_metrics[key])
        std_value = np.std(valid_metrics[key])
        print(' - Validation [%s]: %1.4f +- %1.4f\n' %
            (key.upper(), mean_value, std_value))

    file_path = os.path.join(RESULTS_DIR, 'train_results.pkl')
    with open(file_path, 'wb') as fp:
        pickle.dump(train_metrics, fp)

    file_path = os.path.join(RESULTS_DIR, 'valid_results.pkl')
    with open(file_path, 'wb') as fp:
        pickle.dump(valid_metrics, fp)
        

        

        
        
        
