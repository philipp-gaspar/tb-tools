import sys
import os
import glob
import pickle

import tensorflow as tf
import numpy as np

from utils import create_folder, check_file
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# Define global variables
HOME_DIR = os.environ['HOME']
DATA_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'data-schenzen', 'raw')
PACKAGE_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'tb-tools')

EXPERIMENTS_DIR = os.path.join(PACKAGE_DIR, 'experiments')
CNN_OUTPUTS_DIR = os.path.join(EXPERIMENTS_DIR, 'CNN', 'outputs')

# important for reproducibility
SEED = 13
np.random.seed(SEED)

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
    files_ds = tf.data.Dataset.from_tensor_slices(X)
    ds = files_ds.map(
        lambda file: parse_images(file, width, height, channels))

    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

    return ds

if __name__ == '__main__':
    # -------------------- #
    #    LOAD FILENAMES    #
    # ==================== #
    dataset_name = 'schenzen'
    X, y = load_filenames(DATA_DIR, dataset_name)

    exp_name = '%s-Embeddings' % (dataset_name.title())
    outputs_dir = os.path.join(EXPERIMENTS_DIR, exp_name, 'outputs')
    results_dir = os.path.join(EXPERIMENTS_DIR, exp_name, 'results')
    create_folder(outputs_dir)
    create_folder(results_dir)

    # --------------------------------------- #
    #    STRATIFIED KFOLD CROSS VALIDATION    #
    # ======================================= #
    kfold = StratifiedKFold(n_splits=10, 
                            shuffle=True, 
                            random_state=SEED)

    fold = 0
    for trn_idx, val_idx in kfold.split(X, y):
        print('Fold #%i' % fold)

        X_trn, X_val, y_trn, y_val = perform_xval(X, y, trn_idx, val_idx)

        # --------------------------------------- #
        #    TENSORFLOW BATCH DATASET PIPELINE    #
        # ======================================= #
        train_ds = create_batch_dataset(X_trn, batch_size=64)

        # Load TensorFlow original trained model
        file_name = 'cnn_fold%i.h5' % fold
        input_file = os.path.join(CNN_OUTPUTS_DIR, file_name)
        check_file(input_file)
        orig_model = tf.keras.models.load_model(input_file)

        # Remove dense layer and create embeddings
        model = tf.keras.Model(inputs=orig_model.input, 
                               outputs=orig_model.get_layer('dense').output)

        embeddings = []
        targets = []
        for batch, (image, label) in enumerate(train_ds):
            emb = model(image, training=False)
            embeddings.append(emb)
            targets.append(label)

        # Concatenate Tensors
        embeddings = tf.concat(embeddings, axis=0)
        targets = tf.concat(targets, axis=0)

        # ------------------------------- #
        #    SAVING EMBEDDINGS & MODELS   #
        # =============================== #
        results = {'X': embeddings.numpy(), 
                   'y': targets.numpy()}

        file_path = os.path.join(results_dir, 'embedding_fold%i.pkl' % fold)
        with open(file_path, 'wb') as fp:
            pickle.dump(results, fp)

        file_path = os.path.join(outputs_dir, 'cnn_emb_fold%i.h5' % fold)
        model.save(file_path, save_format='h5')

        # update fold number
        fold += 1
