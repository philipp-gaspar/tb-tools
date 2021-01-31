import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pickle
import glob
import logging

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

import utils as utils
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from collections import Counter, defaultdict
from itertools import product

# Define global variables
try:
    flag = int(os.environ['COLAB'])
    HOME_DIR = "/content/drive/MyDrive/TB-TOOLS"
except:
    flag = 0   
    HOME_DIR = os.environ['HOME']

PACKAGE_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'tb-tools')

EXPERIMENTS_DIR = os.path.join(PACKAGE_DIR, 'experiments')
OUTPUTS_DIR = os.path.join(EXPERIMENTS_DIR, 'SOM', 'outputs')
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'SOM', 'results')
FIGURES_DIR = os.path.join(EXPERIMENTS_DIR, 'SOM', 'figures')
utils.create_folder(OUTPUTS_DIR)
utils.create_folder(RESULTS_DIR)
utils.create_folder(FIGURES_DIR)

# important for reproducibility
SEED = 13
np.random.seed(SEED)

if __name__ == "__main__":
    # -------------------- #
    #    LOAD FILENAMES    #
    # ==================== #
    # Set logger 
    now = datetime.now()
    date = str(now.strftime('%d.%b.%y'))
    log_name = 'SOM_%s.log' % (str(date))
    utils.set_logger(os.path.join(EXPERIMENTS_DIR, 'SOM', log_name))

    # Set labels for targets
    label_names = {0: 'Not TB', 1:'TB'}

    n_folds = 10
    for fold in range(n_folds):
        filename = 'embedding_fold%i.pkl' % fold
        embedding_dir = os.path.join(EXPERIMENTS_DIR, 'Schenzen-Embeddings', 'results')
        input_file = os.path.join(embedding_dir, filename)
        utils.check_file(input_file)
        
        # Open dictionary with embeddings
        with open(input_file, 'rb') as handle:
            embedding = pickle.load(handle)

        # Data normalization
        data = StandardScaler().fit_transform(embedding['X'])
        target = embedding['y']
        
        # --------------- #
        #    SOM MODEL    #
        # =============== #
        logging.info('--- SOM MODEL [Fold #%i] ---' % fold)
        
        n_neurons = 15
        m_neurons = 9
        som = MiniSom(n_neurons, m_neurons, data.shape[1], 
                      sigma=2.0, learning_rate=.5, 
                      neighborhood_function='gaussian', random_seed=SEED)

        som.pca_weights_init(data)
        som.train(data, num_iteration=1000, verbose=True)
        
        # Get winner filenames for each neuron in SOM map
        winner_files = defaultdict(list)
        for entry, file in zip(data, embedding['files']):
            winner_files[som.winner(entry)].append(file)

        # Count the number of events of each class for
        # each neuron in SOM map
        map_results = defaultdict(tuple)
        for position in winner_files:
            win_names = winner_files[position]
            n_H0 = len([i for i in win_names if '_0.png' in i])
            n_H1 = len([i for i in win_names if '_1.png' in i])
            map_results[position] = (n_H0, n_H1)
            logging.info(' - Neuron [%i, %i] = %i/%i events' % \
                (position[0], position[1], n_H0, n_H1))
        logging.info(' ')
                    
        # ------------------------------ #
        #    SAVING RESULTS & OUTPUTS    #
        # ============================== #
        file_path = os.path.join(OUTPUTS_DIR, 'som_fold%i.pkl' % fold)
        with open(file_path, 'wb') as fp:
            pickle.dump(som, fp)

        file_path = os.path.join(RESULTS_DIR, 'winner_files_fold%i.pkl' % fold)
        with open(file_path, 'wb') as fp:
            pickle.dump(winner_files, fp)

        file_path = os.path.join(RESULTS_DIR, 'winner_count_fold%i.pkl' % fold)
        with open(file_path, 'wb') as fp:
            pickle.dump(map_results, fp)

        # ---- FIGURE #1 ---- #
        colors = ['C3', 'C2']
        w_x, w_y = zip(*[som.winner(d) for d in data])
        w_x = np.array(w_x)
        w_y = np.array(w_y)

        plt.figure(figsize=(12, 6))
        plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=0.5)
        plt.colorbar()

        for c in np.unique(target):
            idx_target = target==c
            plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                        w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                        s=50, c=colors[c-1], label=label_names[c])
        plt.legend(loc='upper right', facecolor='w')
        plt.grid()
        output_file = os.path.join(FIGURES_DIR, 'map1_fold%i.png' % fold)
        plt.savefig(output_file, bbox_inches='tight', format='png')

        labels_map = som.labels_map(data, [label_names[t] for t in target])

        # ---- FIGURE #2 ---- #
        fig = plt.figure(figsize=(12, 6))
        the_grid = gridspec.GridSpec(n_neurons, n_neurons, fig)

        for i, j in itertools.product(range(n_neurons), range(m_neurons)):
            if len(labels_map[i, j]) == 0:
                label_frac = [0, 0]
            else:
                label_frac = [labels_map[i,j][l] for l in label_names.values()]
        
            plt.subplot(the_grid[n_neurons - 1 - j, i], aspect=1)
            patches, texts = plt.pie(label_frac, colors=['C2', 'C3'])

        plt.legend(patches, label_names.values(), bbox_to_anchor=(1.5, 1.5), ncol=3)
        output_file = os.path.join(FIGURES_DIR, 'map2_fold%i.png' % fold)
        plt.savefig(output_file, bbox_inches='tight', format='png')