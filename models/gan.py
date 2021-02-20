from tensorflow import keras

loss = 0.0002
momentum = 0.5

# optimizer
optimizer = keras.optimizers.Adam(loss, beta_1 = momentum)

def define_model(disc_model, gen_model):
  # sequential model means stacked layers
  model = keras.models.Sequential()

  disc_model.trainable = False
  
  model.add(gen_model)
  model.add(disc_model)

  model.compile(loss='binary_crossentropy', optimizer=optimizer)
  return model

# print model summary
# define_model(100).summary()