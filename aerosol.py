# lexiboi 10.02.2020
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

data_dir = Path('./data/')

# training data is scanned posters
poster_height = 989 # px
poster_width = 700
poster_channels = 3

# training batch
batch_size = 5

#split dataset for training and validation
validation_split = 0.1


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


