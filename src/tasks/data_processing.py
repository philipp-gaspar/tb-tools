import os
import random
from tensorflow.python.ops.gen_data_flow_ops import record_input_eager_fallback
from tqdm import tqdm
from src.utils.data_utils import check_file
import luigi

from src.murabei_imaging.write_tfrecord import DATA_DIR

import pandas as pd
import numpy as np
import tensorflow as tf

from src.utils.data_utils import (
    setup_murabei_imaging, 
    check_file, 
    _bytes_feature, 
    _int64_feature
)

# --------------------------- #
#    UNPACK ANALYSIS PATHS    #
# =========================== #
paths = setup_murabei_imaging()
MURABEI_DATA_DIR = paths['data']

# ---------------------- #
#    WRITE TF-RECORDS    #
# ====================== #
class WriteTFRecords(luigi.Task):
    data_dir = luigi.Parameter(default=MURABEI_DATA_DIR)
    output_name = luigi.Parameter(default='luigi_murabei_images.tfrec')

    def _build_examples(self, df, seed=42):
        cell_mapper = {'eugenio': 'CelularEugenio', 
                       'simone': 'CelularSimone'}

        examples = []

        for _, row in df.iterrows():
            # get image filepath
            cell_folder = cell_mapper[row['celular']]
            name = row['foto_path']
            filepath = os.path.join(self.data_dir, 'raw', cell_folder, name)

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

    def run(self):
        # access spreadsheet with image metadata
        columns = ['foto_id', 'foto_path', 'id_raiox', 'setting_id', 
               'id_replica', 'celular', 'luz', 'tripe', 'negatoscopio', 
               'centralidade', 'resolucao', 'temporizador', 'app']

        input_filepath = os.path.join(self.data_dir, 'raw', 
            '2020-12-18__desing_experimental_final.xlsx')
        check_file(input_filepath)

        # open spreadsheet as a Pandas Dataframe
        df = pd.read_excel(input_filepath, usecols=columns, engine='openpyxl')
        df.fillna(np.NaN, inplace=True)
        print(' - MetaData Spreadsheet: %s' % input_filepath)

        # remove empty entries on `foto_path` column
        df_small = df[df['celular']=='simone']  # TO DO: EXCLUDE LATER
        df_small = df_small.dropna(subset=['foto_path'])

        # --- WRITE TFRECORD DATA --- #
        examples_list = self._build_examples(df_small)
        output_filepath = os.path.join(self.data_dir, 'interim', self.output_name)

        writer = tf.io.TFRecordWriter(output_filepath)
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

    def output(self):
        output_filepath = os.path.join(self.data_dir, 'interim', self.output_name)
        return luigi.LocalTarget(output_filepath)

# --------------------- #
#    READ TF-RECORDS    #
# ===================== #
class ReadTFRecords(luigi.Task):
    def requires(self):
        return WriteTFRecords()

    def _read_examples(self):

    def run(self):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False # disable order, increase speed
        
        input_filepath = self.input().open().name
        dataset = tf.data.TFRecordDataset(input_filepath)

        # create a dictionary describing the features
        tfrecord_format = {
            'photo_id': tf.io.FixedLenFeature([], tf.int64), 
            'xray_id': tf.io.FixedLenFeature([], tf.int64), 
            'setting_id': tf.io.FixedLenFeature([], tf.int64), 
            'replica_id': tf.io.FixedLenFeature([], tf.int64),
            'cell': tf.io.FixedLenFeature([], tf.string), 
            'image_encoded': tf.io.FixedLenFeature([], tf.string), 
        }


        
        print('=========================\n\n')
        print(dataset)
        print('\n\n===========================')
