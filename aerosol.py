from pathlib import Path
from tensorflow import keras

import models.discriminator

data_dir = Path('./data/')

# poster data
poster_height = 100 # px
poster_width = 70
poster_channels = 3

# training batch
batch_size = 20

#split dataset for training and validation
validation_split = 0.1

# datasets
train_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split = validation_split,
  subset='training',
  seed = 123,
  image_size = (poster_height, poster_width),
  batch_size = batch_size)

validate_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split = validation_split,
  subset='validation',
  seed = 123,
  image_size = (poster_height, poster_width),
  batch_size = batch_size)