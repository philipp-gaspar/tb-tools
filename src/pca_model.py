import sys
import os
import pickle
import glob
import logging

import tensorflow as tf
import numpy as np
import utils as utils

from sklearn.decomposition import PCA
from datetime import datetime
from joblib import dump

# Define global variables
HOME_DIR = os.environ['HOME']
PACKAGE_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'tb-tools')

EXPERIMENTS_DIR = os.path.join(PACKAGE_DIR, 'experiments')
OUTPUTS_DIR = os.path.join(EXPERIMENTS_DIR, 'PCA', 'outputs')
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'PCA', 'results')
utils.create_folder(OUTPUTS_DIR)
utils.create_folder(RESULTS_DIR)

# important for reproducibility
SEED = 13
np.random.seed(SEED)

if __name__ == "__main__":
    # set logger
    now = datetime.now()
    date = str(now.strftime('%d.%b.%y'))
    log_name = 'PCA_%s.log' % (str(date))
    utils.set_logger(os.path.join(EXPERIMENTS_DIR, 'PCA', log_name))

    # Empty dict for results
    results = {'variance': [], 
               'norm_variance': [], 
               'N': []}

    # -------------------- #
    #    LOAD FILENAMES    #
    # ==================== #
    n_folds = 10
    for fold in range(n_folds):
        filename = 'embedding_fold%i.pkl' % fold
        embedding_dir = os.path.join(EXPERIMENTS_DIR, 'Schenzen-Embeddings', 'results')
        input_file = os.path.join(embedding_dir, filename)
        utils.check_file(input_file)
        
        # open dictionary with embeddings
        with open(input_file, 'rb') as handle:
            embedding = pickle.load(handle)
        
        # --------------- #
        #    PCA MODEL    #
        # =============== #
        logging.info('--- PCA MODEL [Fold #%i] ---' % fold)

        n_comp = 2
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_new = pca.fit_transform(embedding['X'])

        variance_ratio = pca.explained_variance_ratio_
        results['norm_variance'].append(variance_ratio)

        variance = pca.explained_variance_
        results['variance'].append(variance)

        logging.info(' - First variance: %1.2f' % variance[0])
        logging.info(' - Second variance: %1.2f' % variance[1])
        logging.info(' - Ratio = %1.2f\n' % (variance[0]/variance[1]))

        logging.info(' - First (normalized) variance: %1.2f' % variance_ratio[0])
        logging.info(' - Second (normalized) variance: %1.2f' % variance_ratio[1])
        logging.info(' - Ratio = %1.2f\n' % (variance_ratio[0]/variance_ratio[1]))

        N = 5 * np.sqrt(len(embedding['y']))
        logging.info(' - N = %1.1f\n' % N)
        results['N'].append(N)

        # -------------------- #
        #    SAVING OUTPUTS    #
        # ==================== #
        file_path = os.path.join(OUTPUTS_DIR, 'pca_fold%i.joblib' % fold)
        dump(pca, file_path)

    # -------------------- #
    #    SAVING RESULTS    #
    # ==================== #
    file_path = os.path.join(RESULTS_DIR, 'train_results.pkl')
    with open(file_path, 'wb') as fp:
        pickle.dump(results, fp)




