from pathlib import Path
from tensorflow import keras

from trainers.dcgan import train

data_dir = Path('./data/training')

trained_model_dir = './models/trained'

# poster data
poster_height = 128 # px
poster_width = 96
poster_channels = 3

# training batch
epochs = 800
batch_size = 20


# datasets
train_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir,
  seed = 123,
  image_size = (poster_height, poster_width),
  batch_size = batch_size)

train(train_ds.cache(), epochs, batch_size)