from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import dropout

# poster data
# x = 100 px
# y = 70
# z = 3 (rgb)
# == (100, 70, 3)

def define_model():

  alpha = 0.2
  dropout = 0.4
  loss = 0.0002
  momentum = 0.5

  # sequential model means stacked layers
  model = keras.models.Sequential()

  # padding convoluting with padding same means original shape is kept
  model.add(layers.Conv2D(64, (3, 3), padding = 'same', input_shape = (100, 70, 3)))
  model.add(layers.LeakyReLU(alpha = alpha))

  model.add(layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(layers.LeakyReLU(alpha = alpha))

  model.add(layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(layers.LeakyReLU(alpha = alpha))

  model.add(layers.Conv2D(256, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(layers.LeakyReLU(alpha = alpha))

  model.add(layers.Flatten())
  model.add(layers.Dropout(0.4))
  model.add(layers.Dense(1, activation = 'sigmoid'))


  # optimizer
  optimizer = keras.optimizers.Adam(loss, beta_1 = momentum)

  # compile
  model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

  return model