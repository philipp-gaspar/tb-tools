import os
import tensorflow as tf
import numpy as np


from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_curve

# ===================== #
#    KERAS CALLBACKS    #
# ===================== #
class EarlyStoppingAtSP(Callback):
    """
    Stop training when the SP indes is at its maximum value.
    """
    def __init__(self, validation_data, patience=3):
        self.patience = patience
        self.best_sp = 0.0
        self.best_weights = None
        self.validation_data = validation_data
        self.best_epoch = 0

    def on_train_begin(self, logs=None):
        # the number of epochs it has waited when SP is no longer maximum
        self.wait = 0
        # the epoch training stops at
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        y_true = self.validation_data[1]
        y_pred = self.model.predict(self.validation_data[0])

        # compute SP index
        fa, pd, _ = roc_curve(y_true, y_pred)
        sp = np.sqrt(np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))
        knee = np.argmax(sp)

        logs['sp_val'] = sp[knee]

        if sp[knee] > self.best_sp:
            self.best_sp = sp[knee]
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_label(file_path):
    name = 'TB'
    class_name = file_path.split(os.path.sep)[-2]
    label = int(name == class_name)

    return label

def parse_images(filename, width, height, channels):
    """
    Reads an image from a file, decodes it into a dense tensor, 
    and resizes it to a fixed shape.
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
    
    # add batch dimension
    #image = tf.expand_dims(image, axis=0)

    return image, label
