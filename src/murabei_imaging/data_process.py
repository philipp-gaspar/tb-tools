import sys
import os

import pandas as pd
import numpy as np
import tensorflow as tf

from src.utils import setup_murabei_imaging

if __name__ == '__main__':
    # --------------------------- #
    #    UNPACK ANALYSIS PATHS    #
    # =========================== #
    paths = setup_murabei_imaging(mode='local')
    data_dir = paths['data']
    experiments_dir = paths['experiments']
    outputs_dir = paths['outputs']
    results_dir = paths['results']

    usecols = ['foto_id', 'foto_path', 'id_raiox', 'setting_id', 
               'id_replica', 'celular', 'luz', 'tripe', 'negatoscopio', 
               'centralidade', 'resolucao', 'temporizador', 'app']
    filename = os.path.join(data_dir, 'raw', \
        '2020-12-18_desing_experimental.csv')
    df = pd.read_csv(filename, usecols=usecols)
    df.fillna(np.NaN, inplace=True)

    # --------------------- #
    #    DATA DICTIONARY    #
    # --------------------- #
    cell_mapper = {'eugenio': 'CelularEugenio', 
                  'simone': 'CelularSimone'}

    # create metadata dictionary
    metadata = {}
    usecols.remove('foto_id')

    for index, row in df.iterrows():
        key1 = row['foto_id']
        metadata[key1] = {} 

        for col in usecols:
            value = row[col]
            metadata[key1][col] = value

        # get image filename
        cell = metadata[key1]['celular']
        cell = cell_mapper[cell]
        name = metadata[key1]['foto_path']

        if type(name) is float:
            image = None
        else:
            filename = os.path.join(data_dir, 'raw', cell, name)

        if os.path.isfile(filename):
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image)
            image = tf.image.convert_image_dtype(image, tf.float32)
        else:
            image = None

        metadata[key1]['data'] = image

    # ----------------- #
    #    SAVE OUTPUT    #
    # ================= #
    

        


    