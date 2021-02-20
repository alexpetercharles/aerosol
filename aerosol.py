from pathlib import Path
from tensorflow import keras

from models import gan, discriminator, generator

from trainers.dcgan import train

data_dir = Path('./data/training')

trained_model_dir = './models/trained'

# poster data
poster_height = 128 # px
poster_width = 96
poster_channels = 3

# training batch
epochs = 5000
# since discriminator is trained two times this is actually half the batch size
batch_size = 5

# input dimensions
latent_dim = 100


# datasets
train_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir,
  seed = 123,
  image_size = (poster_height, poster_width),
  batch_size = batch_size)

# create models
disc_model = discriminator.define_model()
gen_model = generator.define_model(latent_dim)

gan_model = gan.define_model(disc_model, gen_model)
gan_model.summary()

train(gan_model, disc_model, gen_model, train_ds.cache(), epochs, batch_size, latent_dim)