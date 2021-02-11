# lexiboi 10.02.2020
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow import keras

from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Sequential

data_dir = Path('./data/')

# training data is scanned posters
poster_height = 100 # px
poster_width = 70
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

# model input noise
input_shape = 100
noise = Input(shape = (input_shape))

def make_generator_model():
  model = Sequential()

  model.add(Dense(10*7*30, activation = "relu", input_shape = (input_shape,)))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Reshape((10, 7, 30)))
  assert model.output_shape == (None, 10, 7, 30)
  
  model.add(Conv2DTranspose(70, (10, 7), strides = (2, 2), padding = "same", use_bias = False))
  assert model.output_shape == (None, 20, 14, 70)
  model.add(BatchNormalization())
  model.add(LeakyReLU())

  model.add(Conv2DTranspose(3, (10, 7), strides = (5, 5), padding = "same", use_bias = False, activation = 'tanh'))
  assert model.output_shape == (None, 100, 70, 3)

  return model

generator = make_generator_model()

generated_image = generator(tf.random.normal([1, 100]), training = False)

plt.imshow(generated_image[0, :, :, 0])
plt.show()