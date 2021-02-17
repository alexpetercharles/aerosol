from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import dropout

# poster data
# x = 128 px
# y = 96
# z = 3 (rgb)
# == (100, 70, 3)

alpha = 0.2
dropout = 0.4
loss = 0.0002
momentum = 0.5

# optimizer
optimizer = keras.optimizers.Adam(loss, beta_1 = momentum)

def define_model(latent_dim):

  foundation = 256 * 3 * 4

  # sequential model means stacked layers
  model = keras.models.Sequential()

  model.add(layers.Dense(foundation, input_dim = latent_dim))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  model.add(layers.Reshape((4, 3, 256)))
  # (4, 3, 256)

  model.add(layers.Conv2DTranspose(256, (4, 4), strides = (2, 2), padding = 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (8, 6, 256)

  model.add(layers.Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (16, 12, 128)

  model.add(layers.Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (32, 24, 128)

  model.add(layers.Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (64, 48, 128)

  model.add(layers.Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (128, 96, 128)

  model.add(layers.Conv2D(3, (3, 3), activation= 'tanh', padding = 'same'))

  return model

# print model summary
# define_model(100).summary()