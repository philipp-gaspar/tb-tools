import sys
import os
import pickle
import glob

import tensorflow as tf
import numpy as np

from utils import parse_images, create_folder, load_filenames, check_file
from sklearn.decomposition import PCA

# Define global variables
HOME_DIR = os.environ['HOME']
PACKAGE_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'tb-tools')

EXPERIMENTS_DIR = os.path.join(PACKAGE_DIR, 'experiments')
OUTPUTS_DIR = os.path.join(EXPERIMENTS_DIR, 'PCA', 'outputs')
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'PCA', 'results')
create_folder(OUTPUTS_DIR)
create_folder(RESULTS_DIR)

# important for reproducibility
SEED = 13
np.random.seed(SEED)

if __name__ == "__main__":
    # -------------------- #
    #    LOAD FILENAMES    #
    # ==================== #
    n_folds = 10
    for fold in range(n_folds):
        print('Fold #%i' % fold)
        filename = 'embedding_fold%i.pkl' % fold
        embedding_dir = os.path.join(EXPERIMENTS_DIR, 'Schenzen-Embeddings', 'results')
        input_file = os.path.join(embedding_dir, filename)
        check_file(input_file)
        
        # open dictionary with embeddings
        with open(input_file, 'rb') as handle:
            embedding = pickle.load(handle)
        
        # --------------- #
        #    PCA MODEL    #
        # =============== #
        print('-- PCA MODEL ---')
        n_comp = 2
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_new = pca.fit_transform(embedding['X'])

        variance_ratio = pca.explained_variance_ratio_
        variance = pca.explained_variance_

        print('\n - Auxiliary Info to Build SOM')
        print('   First variance: %1.2f' % variance[0])
        print('   Second variance: %1.2f' % variance[1])
        print('   Ratio = %1.2f\n' % (variance[0]/variance[1]))

        print('   First (normalized) variance: %1.2f' % variance_ratio[0])
        print('   Second (normalized) variance: %1.2f' % variance_ratio[1])
        print('   Ratio = %1.2f\n' % (variance_ratio[0]/variance_ratio[1]))


        N = 5 * np.sqrt(len(embedding['y']))
        print('   N = %1.1f\n' % N)



