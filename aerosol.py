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
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dropout, Flatten
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

# models
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

def make_discriminator_model():
  model = Sequential()
  keras.layers.experimental.preprocessing.Rescaling(1./127.5, -1)
  model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                   input_shape=[100, 70, 3]))
  model.add(LeakyReLU())
  model.add(Dropout(0.3))

  model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(LeakyReLU())
  model.add(Dropout(0.3))

  model.add(Flatten())
  model.add(Dense(1))

  return model

# loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

# optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# create models
generator = make_generator_model()
discriminator = make_discriminator_model()

# training checkpoints
checkpoint_dir = './training'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# training loop
epochs = 50
examples_to_generate = 16

seed = tf.random.normal([examples_to_generate, input_shape])

@tf.function
def train_step(posters): 
  noise = tf.random.normal([batch_size, input_shape])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(posters, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    
    for image_batch in dataset:
      train_step(image_batch[0])

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 500 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(configure_for_performance(train_ds), 1000000000)