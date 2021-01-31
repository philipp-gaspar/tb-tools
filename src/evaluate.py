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
from utils import perform_xval, create_batch_dataset
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Define global variables
try:
    flag = int(os.environ['COLAB'])
    HOME_DIR = "/content/drive/MyDrive/TB-TOOLS"
    DATA_DIR = os.path.join("/content/drive/My Drive/BRICS - TB Latente/Dados/ChinaSet_AllFiles/CXR_png")
except:
    flag = 0   
    HOME_DIR = os.environ['HOME']
    DATA_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'data-schenzen', 'raw') 
         
PACKAGE_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'tb-tools')

EXPERIMENTS_DIR = os.path.join(PACKAGE_DIR, 'experiments')
OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, 'CNN', 'outputs')
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'CNN', 'results')
EVALUATIONS_DIR = os.path.join(EXPERIMENTS_DIR, 'CNN', 'evaluations')
create_folder(EVALUATIONS_DIR)
create_folder(OUTPUT_DIR)
create_folder(RESULTS_DIR)

width = 128
height = 128
channels = 1
BATCH_SIZE = 64


#getting all data
input_files = dict()
file_name = 'CHNCXR_*_0.png'
input_files['H0'] = glob.glob(os.path.join(DATA_DIR, file_name))
file_name = 'CHNCXR_*_1.png'
input_files['H1'] = glob.glob(os.path.join(DATA_DIR, file_name))

n_H0 = len(input_files['H0'])
n_H1 = len(input_files['H1'])

X = np.asarray(input_files['H0'] + input_files['H1'])
y = np.concatenate((np.zeros(n_H0), np.ones(n_H1)))

files_ds = tf.data.Dataset.from_tensor_slices(X)

eval_ds = files_ds.map(
            lambda file: parse_images(file, width, height, channels))
eval_ds = eval_ds.batch(BATCH_SIZE)

filename = 'cnn_fold*.h5'

MODELS = glob.glob(os.path.join(OUTPUT_DIR, filename))

for i in range(1, len(MODELS)+1):
    evaluate_metrics = {}
    print('MODEL '+str(i)+'/10')
    #evaluation metrics

    #loading the model
    loaded_model = tf.keras.models.load_model(MODELS[i])
    loaded_model.build()

    #evaluating the model
    eval_loss, eval_acc = loaded_model.evaluate(eval_ds)

    #calculating the sp index
    y_prob_val = loaded_model.predict(eval_ds)
    fpr, tpr, threshold = roc_curve(y, y_prob_val)
    val_auc = auc(fpr, tpr)
    evaluate_metrics['auc'] = (val_auc)

    val_sp = np.sqrt(np.sqrt(tpr*(1-fpr)) * (0.5*(tpr+(1-fpr))))
    val_knee = np.argmax(val_sp)
    evaluate_metrics['sp'] = (val_sp[val_knee])

    print('SP: ' + str(val_sp[val_knee]))

    y_pred_val = np.zeros((len(y_prob_val), 1))
    y_pred_val[y_prob_val > threshold[val_knee]] = 1.0
    conf_matrix = confusion_matrix(y, y_pred_val)
    tn, fp, fn, tp = conf_matrix.ravel()

    #appending to metrics dict
    evaluate_metrics['acc'] = eval_acc
    evaluate_metrics['loss'] = eval_loss
    evaluate_metrics['tpr'] = tp/(tp+fn)
    evaluate_metrics['tnr'] = tn/(tn+fp)
    evaluate_metrics['fpr'] = fp/(fp+tn)
    evaluate_metrics['fnr'] = fn/(tp+fn)

    #confusion matrix
    evaluate_metrics['tp'] = tp
    evaluate_metrics['tn'] = tn
    evaluate_metrics['fp'] = fp
    evaluate_metrics['fn'] = fn

    print('tn: '+str(tn), 'fp: '+str(fp), 'fn: '+str(fn),'tp: '+str(tp) )
    print('tpr: '+str(tp/(tp+fn)), 'tnr: '+str(tn/(tn+fp)), 'fpr: '+str(fp/(fp+tn)),'fnr: '+str(fn/(tp+fn)))
    print('ACCURACY: '+str((tp+tn)/(tp+tn+fp+fn))+'\n')

    #saving results
    file_path = os.path.join(EVALUATIONS_DIR, 'eval{}.pkl'.format(i))
    with open(file_path, 'wb') as fp:
        pickle.dump(evaluate_metrics, fp)