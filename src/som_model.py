import sys
import os
import pickle
import glob

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

from utils import parse_images, create_folder, load_filenames, check_file
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler

# Define global variables
HOME_DIR = os.environ['HOME']
PACKAGE_DIR = os.path.join(HOME_DIR, 'BRICS-TB', 'tb-tools')

EXPERIMENTS_DIR = os.path.join(PACKAGE_DIR, 'experiments')
OUTPUTS_DIR = os.path.join(EXPERIMENTS_DIR, 'SOM', 'outputs')
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'SOM', 'results')
FIGURES_DIR = os.path.join(EXPERIMENTS_DIR, 'SOM', 'figures')
create_folder(OUTPUTS_DIR)
create_folder(RESULTS_DIR)
create_folder(FIGURES_DIR)

# important for reproducibility
SEED = 13
np.random.seed(SEED)

if __name__ == "__main__":
    # -------------------- #
    #    LOAD FILENAMES    #
    # ==================== #
    label_names = {0: 'Not TB', 1:'TB'}

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

        # data normalization
        data = StandardScaler().fit_transform(embedding['X'])
        target = embedding['y']
        
        # --------------- #
        #    SOM MODEL    #
        # =============== #
        print('-- SOM MODEL ---')
        n_neurons = 15
        m_neurons = 9

        som = MiniSom(n_neurons, m_neurons, data.shape[1], 
                      sigma=2.0, learning_rate=.5, 
                      neighborhood_function='gaussian', random_seed=SEED)

        som.pca_weights_init(data)
        som.train(data, 1000, verbose=True)

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
                


        


