import os
import sys
import tensorflow as tf
import numpy as np
import glob
import logging

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_curve
from collections import Counter

# --------------------- #
#    KERAS CALLBACKS    #
# ===================== #
class EarlyStoppingAtSP(Callback):
    """
    Stop training when the SP indes is at its maximum value.
    """
    def __init__(self, validation_data, patience=3):
        self.patience = patience
        self.best_sp = 0.0
        self.best_weights = None
        self.validation_data = validation_data
        self.best_epoch = 0

    def on_train_begin(self, logs=None):
        # the number of epochs it has waited when SP is no longer maximum
        self.wait = 0
        # the epoch training stops at
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        y_true = self.validation_data[1]
        y_pred = self.model.predict(self.validation_data[0])

        # compute SP index
        fa, pd, _ = roc_curve(y_true, y_pred)
        sp = np.sqrt(np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))
        knee = np.argmax(sp)

        logs['sp_val'] = sp[knee]

        if sp[knee] > self.best_sp:
            self.best_sp = sp[knee]
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

# ---------------------- #
#    LOGGER FUNCTIONS    #
# ====================== #
def set_logger(log_path, mode='w'):
    """
    Sets the logger to log info in terminal and in the file 'log_path'.
    In general, it is useful to have a logger so that every output
    to the terminal is saved in a permanent file.

    Example:
    -------
        logging.info('Start training...')
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode=mode)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

# ------------------ #
#    DATA HELPERS    #
# ================== #
def perform_xval(X, y, trn_idx, val_idx, verbose=True):
    """
    Create X and y vectors for training and validation.
    """
    np.random.shuffle(trn_idx)
    np.random.shuffle(val_idx)

    X_trn, X_val = X[trn_idx], X[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]

    if verbose:
        trn_cnt = Counter(y_trn)
        val_cnt = Counter(y_val)
        print(' - Train: %i/%i' % (trn_cnt[0.0], trn_cnt[1.0]))
        print(' - Valid: %i/%i\n' % (val_cnt[0.0], val_cnt[1.0]))

    return X_trn, X_val, y_trn, y_val

def load_filenames(data_dir, dataset_name):
    """
    Read png data and return X and y vectors for model training.
    """
    input_files = dict()

    if dataset_name == 'schenzen':
        file_name = 'CHNCXR_*_0.png'
        input_files['H0'] = glob.glob(os.path.join(data_dir, file_name))
        file_name = 'CHNCXR_*_1.png'
        input_files['H1'] = glob.glob(os.path.join(data_dir, file_name))

        n_H0 = len(input_files['H0'])
        n_H1 = len(input_files['H1'])

        X = np.asarray(input_files['H0'] + input_files['H1'])
        y = np.concatenate((np.zeros(n_H0), np.ones(n_H1)))
    else:
        print('Error! Not valid dataset name.')
        sys.exit()

    return X, y

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_file(input_file):
    try:
        assert os.path.isfile(input_file)
    except AssertionError:
        print('FILE NOT FOUND!')
        print('%s' % input_file)
        sys.exit()

def get_label(file_path):
    name = 'TB'
    class_name = file_path.split(os.path.sep)[-2]
    label = int(name == class_name)

    return label

def parse_images(filename, width, height, channels):
    """
    Reads an image from a PNG file, decodes it into a dense tensor, 
    and resizes it to a fixed shape.

    NOTE: Only tested for Schenzen images.
    """
    parts = tf.strings.split(filename, os.sep)
    name = parts[-1]
    label = tf.strings.substr(name, pos=-5, len=1)
    label = tf.strings.to_number(label, 
        out_type=tf.dtypes.int32)
    
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [width, height])

    return image, label

def create_batch_dataset(X, batch_size, width=128, height=128, channels=1):
    """
    Get PNG images and convert it to a TensorFlow Batch Dataset.

    NOTE: Only tested for Schenzen images.
    """
    file_ds = tf.data.Dataset.from_tensor_slices(X)
    ds = file_ds.map(
        lambda file: parse_images(file, width, height, channels))

    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

    return ds
