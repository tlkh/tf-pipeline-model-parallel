import time
import numpy as np
import tensorflow as tf

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

        
def normalize(ndarray):
    ndarray = ndarray.astype("float32")
    ndarray = (ndarray/127.5) - 1
    return ndarray

