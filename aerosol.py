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

# from models import discriminator
# discriminator_model = discriminator.define_model()
# discriminator_model.summary()

# from utils.image import normalize
# for batch in train_ds:
#  posters, _ = batch
#  print(normalize(posters))

latent_dim = 100

from numpy.random import randn
# generate random input for generator
def generate_latent_points(latent_dim, n_samples):
  # generate points in the latent space
  rand_input = randn(latent_dim * n_samples)
  # reshape for model input
  rand_input = rand_input.reshape(n_samples, latent_dim)
  return rand_input

# plot the posters to verify the posters
# from models import generator
# generator_model = generator.define_model(latent_dim)
# rand_generator_input = generate_latent_points(latent_dim, 20)
# generated = generator_model.predict(rand_generator_input)
# from utils.image import plot_posters
# from utils.image import normalize_to_float
# plot_posters(normalize_to_float(generated))
# 
# # plot the posters to verify the posters
# from utils.image import plot_posters, normalize_to_float, normalize_for_model
# for batch in train_ds:
#  posters, _ = batch   
#  plot_posters(normalize_to_float(normalize_for_model(posters.numpy())))