import sys
import os

import tensorflow as tf

# ------------------- #
#    SETUP HELPERS    #
# =================== #
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

def setup_murabei_imaging(mode='local'):
    """
    Define path variables for Imaging Analysis.
    """
    paths = {}
    if mode == 'local':
        paths['home'] = os.environ['HOME']
        paths['data'] = os.path.join(paths['home'], 'BRICS-TB', 'data-imaging')
        paths['package'] = os.path.join(paths['home'], 'BRICS-TB', 'tb-tools')

        paths['experiments'] = os.path.join(paths['package'], 'experiments')
        paths['outputs'] = os.path.join(paths['experiments'], 'IMG', 'outputs')
        paths['results'] = os.path.join(paths['experiments'], 'IMG', 'results')
        
        create_folder(paths['outputs'])
        create_folder(paths['results'])

    return paths

# ----------------------- #
#    TF-RECORD HELPERS    #
# ======================= #
# The following functions can be used to convert a value to a type
# compatible with `tf.train.Example`
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
