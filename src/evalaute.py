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

try:
    flag = int(os.environ['COLAB'])
    sys.path.append('/content/drive/My Drive/BRICS - TB Latente/Sampling')
except:
    flag = 0    

from utils import parse_images, EarlyStoppingAtSP, create_folder, load_filenames
from utils import perform_xval, create_batch_dataset
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc

if flag:
    HOME_DIR = "/content/drive/My Drive/BRICS - TB Latente/Sampling/"
    DATA_DIR = os.path.join("/content/drive/My Drive/BRICS - TB Latente/Dados/ChinaSet_AllFiles/CXR_png")
else:
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