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

  foundation = 128 * 3 * 4

  # sequential model means stacked layers
  model = keras.models.Sequential()

  model.add(layers.Dense(foundation, input_dim = latent_dim))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  model.add(layers.Reshape((4, 3, 128)))
  # (4, 3, 128)

  model.add(layers.Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (8, 6, 128)

  model.add(layers.Conv2DTranspose(64, (4, 4), strides = (2, 2), padding = 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (16, 12, 64)

  model.add(layers.Conv2DTranspose(64, (4, 4), strides = (2, 2), padding = 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (32, 24, 64)

  model.add(layers.Conv2DTranspose(64, (4, 4), strides = (2, 2), padding = 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (64, 48, 64)

  model.add(layers.Conv2DTranspose(64, (4, 4), strides = (2, 2), padding = 'same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (128, 96, 64)

  model.add(layers.Conv2D(3, (3, 3), activation= 'tanh', padding = 'same'))

  # model.compile(optimizer=optimizer)

  return model

# print model summary
# define_model(100).summary()