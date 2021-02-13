from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import dropout

# poster data
# x = 128 px
# y = 96
# z = 3 (rgb)
# == (100, 70, 3)

def define_model():

  alpha = 0.2
  dropout = 0.4
  loss = 0.0002
  momentum = 0.5

  # sequential model means stacked layers
  model = keras.models.Sequential()

  # padding same keeps shape, strides 2 is best practise
  model.add(layers.Conv2D(64, (3, 3), padding = 'same', input_shape = (128, 96, 3)))
  model.add(layers.LeakyReLU(alpha = alpha))
  # (128, 96, 6)

  model.add(layers.Conv2D(64, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(layers.LeakyReLU(alpha = alpha))
  # (64, 48, 64)

  model.add(layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(layers.LeakyReLU(alpha = alpha))
  # (32, 24, 128)

  model.add(layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(layers.LeakyReLU(alpha = alpha))
  # (16, 12, 128)

  model.add(layers.Conv2D(256, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(layers.LeakyReLU(alpha = alpha))
  # (8, 6, 256)

  model.add(layers.Conv2D(256, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(layers.LeakyReLU(alpha = alpha))
  # (4, 3, 256)

  model.add(layers.Flatten())
  # (3072)

  model.add(layers.Dropout(dropout))
  model.add(layers.Dense(1, activation = 'sigmoid'))
  # (1)


  # optimizer
  optimizer = keras.optimizers.Adam(loss, beta_1 = momentum)

  # compile
  model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

  return model

# print model summary
define_model().summary()