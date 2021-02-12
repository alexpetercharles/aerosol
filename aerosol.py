# lexiboi 10.02.2020
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import time

from pathlib import Path
from tensorflow import keras

from IPython import display

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose, Conv2D
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dropout, Flatten, ZeroPadding2D
from tensorflow.python.framework import dtypes

data_dir = Path('./data/')

# training data is scanned posters
poster_height = 100 # px
poster_width = 70
poster_channels = 3

# training batch
batch_size = 20

#split dataset for training and validation
validation_split = 0.01

# datasets
train_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split = validation_split,
  subset="training",
  seed = 123,
  image_size = (poster_height, poster_width),
  batch_size = batch_size)

validate_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split = validation_split,
  subset="validation",
  seed = 123,
  image_size = (poster_height, poster_width),
  batch_size = batch_size)

# model input noise
input_shape = 100