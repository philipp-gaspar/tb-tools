import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import glob
import pickle

import tensorflow as tf
import numpy as np

import utils as utils
from collections import Counter
from sklearn.model_selection import StratifiedKFold

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
CNN_OUTPUTS_DIR = os.path.join(EXPERIMENTS_DIR, 'CNN', 'outputs')

# important for reproducibility
SEED = 13
np.random.seed(SEED)

if __name__ == '__main__':
    # -------------------- #
    #    LOAD FILENAMES    #
    # ==================== #
    dataset_name = 'schenzen'
    X, y = utils.load_filenames(DATA_DIR, dataset_name)

    exp_name = '%s-Embeddings' % (dataset_name.title())
    outputs_dir = os.path.join(EXPERIMENTS_DIR, exp_name, 'outputs')
    results_dir = os.path.join(EXPERIMENTS_DIR, exp_name, 'results')
    utils.create_folder(outputs_dir)
    utils.create_folder(results_dir)

    # --------------------------------------- #
    #    STRATIFIED KFOLD CROSS VALIDATION    #
    # ======================================= #
    kfold = StratifiedKFold(n_splits=10, 
                            shuffle=True, 
                            random_state=SEED)

    fold = 0
    for trn_idx, val_idx in kfold.split(X, y):
        print('Fold #%i' % fold)

        X_trn, X_val, y_trn, y_val = utils.perform_xval(X, y, trn_idx, val_idx)

        # --------------------------------------- #
        #    TENSORFLOW BATCH DATASET PIPELINE    #
        # ======================================= #
        train_ds = utils.create_batch_dataset(X_trn, batch_size=64)

        # Load TensorFlow original trained model
        file_name = 'cnn_fold%i.h5' % fold
        input_file = os.path.join(CNN_OUTPUTS_DIR, file_name)
        utils.check_file(input_file)
        orig_model = tf.keras.models.load_model(input_file, compile=False)

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

        # Get file names
        file_names = [file.split('/')[-1] for file in X_trn]

        # ------------------------------- #
        #    SAVING EMBEDDINGS & MODELS   #
        # =============================== #
        results = {'X': embeddings.numpy(), 
                   'y': targets.numpy(), 
                   'files': file_names}

        file_path = os.path.join(results_dir, 'embedding_fold%i.pkl' % fold)
        with open(file_path, 'wb') as fp:
            pickle.dump(results, fp)

        file_path = os.path.join(outputs_dir, 'cnn_emb_fold%i.h5' % fold)
        model.save(file_path, save_format='h5')

        # update fold number
        fold += 1
