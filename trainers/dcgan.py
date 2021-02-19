import os
import time
import tensorflow
from utils.image import normalize_for_model, save_posters

checkpoint_dir = './models/trained/checkpoints'

save_checkpoint_epoch = 50
save_image_epoch = 10

# input dimensions
latent_dim = 100

cross_entropy = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)

@tensorflow.function
def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tensorflow.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tensorflow.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return cross_entropy(tensorflow.ones_like(fake_output), fake_output)

@tensorflow.function
def train_step(generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    batch_size,
    posters):

  noise = tensorflow.random.normal([batch_size, latent_dim])
  with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:

    generated_posters = generator(noise, training = True)

    disc_real_out = discriminator(posters, training = True)
    disc_fake_out = discriminator(generated_posters, training = True)

    gen_loss = generator_loss(disc_fake_out)
    disc_loss = discriminator_loss(disc_real_out, disc_fake_out)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(train_dataset, epochs, batch_size):
  from models import generator, discriminator

  generator_model = generator.define_model(latent_dim)
  discriminator_model = discriminator.define_model()

  generator_optimizer = generator.optimizer
  discriminator_optimizer = discriminator.optimizer

  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tensorflow.train.Checkpoint(generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator_model,
    discriminator=discriminator_model)

  checkpoint.restore(tensorflow.train.latest_checkpoint(checkpoint_dir))

  seed = tensorflow.random.normal([25, latent_dim])
  save_posters(generator_model(seed, training = False).numpy(), 0)

  for epoch in range(epochs):
    start = time.time()

    train_dataset = train_dataset.shuffle(100, reshuffle_each_iteration = True)
    
    for batch in train_dataset:
      posters, _ = batch

      train_step(generator_model, 
      discriminator_model, 
      generator_optimizer,
      discriminator_optimizer,
      batch_size,
      normalize_for_model(posters))

      if (epoch + 1) % save_checkpoint_epoch == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

      if (epoch + 1) % save_image_epoch == 0:
        save_posters(generator_model(seed, training = False).numpy(), epoch)
    
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
