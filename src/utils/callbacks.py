# %% Packages

import os
import tensorflow as tf
from typing import Tuple
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# %% Callbacks


# Loading callbacks
def load_callbacks(
    name: str, monitor: str, patience: int
) -> Tuple[TensorBoard, EarlyStopping]:
    tensorboard = load_tensorboard(name=name)
    earlystopping = load_earlystopping(monitor=monitor, patience=patience)
    return (tensorboard, earlystopping)


# Tensorboard
def load_tensorboard(name: str) -> TensorBoard:
    log_dir = os.path.join("logs", "tensorboard", name)
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)


# Early stopping
def load_earlystopping(monitor: str, patience: int) -> EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)
