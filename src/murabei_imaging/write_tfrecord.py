import os
import random

import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from src.utils.data_utils import (
    setup_murabei_imaging, check_file, 
    _bytes_feature, _float_feature, _int64_feature)

# --------------------------- #
#    UNPACK ANALYSIS PATHS    #
# =========================== #
paths = setup_murabei_imaging()
DATA_DIR = paths['data']
EXPERIMENTS_DIR = paths['experiments']
OUTPUTS_DIR = paths['outputs']
RESULTS_DIR = paths['results']

def _build_examples(df, seed=42):
    # mapper dictionary from entry/folder
    cell_mapper = {'eugenio': 'CelularEugenio', 
                   'simone': 'CelularSimone'}

    examples = []

    for index, row in df.iterrows():
        # get image file path
        cell_folder = cell_mapper[row['celular']]
        name = row['foto_path']
        filepath = os.path.join(DATA_DIR, 'raw', cell_folder, name)

        metadata = {
            'photo_id': int(row['foto_id']), 
            'xray_id': int(row['id_raiox']), 
            'setting_id': int(row['setting_id']), 
            'replica_id': int(row['id_replica']), 
            'cell': row['celular'].encode(), 
            'filepath': filepath.encode(),
        }

        examples.append(metadata)

    random.seed(seed)
    random.shuffle(examples)

    return examples

if __name__ == '__main__':
    print('-----------------------------')
    print('--- START DATA PROCESSING ---')
    print('-----------------------------')
    
    # access spreadsheet with image metadata
    columns = ['foto_id', 'foto_path', 'id_raiox', 'setting_id', 
               'id_replica', 'celular', 'luz', 'tripe', 'negatoscopio', 
               'centralidade', 'resolucao', 'temporizador', 'app']
    filename = os.path.join(DATA_DIR, 'raw', '2020-12-18__desing_experimental_final.xlsx')
    check_file(filename)

    # open spreadsheet as a Pandas dataframe
    df = pd.read_excel(filename, usecols=columns, engine='openpyxl')
    df.fillna(np.NaN, inplace=True)
    print(' - Metadata Spreadsheet: %s' % filename)

    # remove empty entries on `foto_path` column
    df_small = df[df['celular']=='simone']  # TO DO: EXCLUDE LATER
    df_small = df_small.dropna(subset=['foto_path'])

    # ------------------------- #
    #    WRITE TFRECORD DATA    #
    # ========================= #
    examples_list = _build_examples(df_small)
    output_filename = os.path.join(DATA_DIR, 'interim', 'murabei_images.tfrec')
    
    writer = tf.io.TFRecordWriter(output_filename)
    for example in tqdm(examples_list): 
        if os.path.isfile(example['filepath']):
            image_encoded = tf.io.read_file(example['filepath'])
            
            feature = {
                'photo_id': _int64_feature(example['photo_id']), 
                'xray_id': _int64_feature(example['xray_id']),
                'setting_id': _int64_feature(example['setting_id']), 
                'replica_id': _int64_feature(example['replica_id']), 
                'cell': _bytes_feature(example['cell']), 
                'image_encoded': _bytes_feature(image_encoded)
            }

            tf_example = tf.train.Example(
                features=tf.train.Features(feature=feature))

            writer.write(tf_example.SerializeToString())
        else:
            continue

    writer.close()

    print('---------------------------')
    print('--- END DATA PROCESSING ---')
    print('---------------------------')

