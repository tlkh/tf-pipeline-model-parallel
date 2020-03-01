import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode",
                    help="Fine-tuning mode: one of [`pipeline`, `normal`, `none`]")
args = parser.parse_args()

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import models
import utils

if args.mode == "pipeline":
    tf.config.experimental.set_synchronous_execution(False)
tf.config.threading.set_inter_op_parallelism_threads(40)
tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

BATCH_SIZE = 320

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
num_classes = np.max(y_train) + 1
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
x_train = utils.normalize(x_train)
num_train = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.prefetch(64)
_ = str(train_dataset.take(1))

steps_per_epoch = int(num_train / BATCH_SIZE / 2) + 1

if args.mode == "pipeline":
    model = models.PipelineCNN(num_classes=num_classes)
elif args.mode == "normal":
    model = models.ParallelCNN(num_classes=num_classes)
elif args.mode == "none":
    model = models.SingleCNN(num_classes=num_classes)
    
opt = tf.keras.optimizers.Adam()
opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
model.compile(loss="categorical_crossentropy",
              optimizer=opt)

time_history = utils.TimeHistory()

train_log = model.fit(train_dataset, steps_per_epoch=steps_per_epoch,
                      epochs=3, verbose=1,
                      callbacks=[time_history])

peak_fps = int(steps_per_epoch*BATCH_SIZE/min(time_history.times))

print("* Params:", model.count_params())
print("* Peak FPS:", peak_fps)
