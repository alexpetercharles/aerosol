from numbers import Number
import os
import time
import tensorflow
from utils.image import normalize_for_model, save_posters

checkpoint_dir = './models/trained/checkpoints'

save_checkpoint_epoch = 10
save_image_epoch = 5

def sum_performance(batch, discriminator, generator, latent_dim, batch_size):
  real_posters, _ = batch
  fake_posters = generator.predict(tensorflow.random.normal([batch_size, latent_dim]))
  real_labels = tensorflow.ones((batch_size, 1))
  fake_labels = tensorflow.zeros((batch_size, 1))
  
  _, acc_real = discriminator.evaluate(normalize_for_model(real_posters), real_labels, verbose = 0)
  _, acc_fake = discriminator.evaluate(fake_posters, fake_labels, verbose = 0)
  print('Accuracy real: ', (acc_real*100), 'fake: ', (acc_fake*100))

def train(gan, discriminator, generator, dataset, epochs, batch_size, latent_dim):
  # checkpoint saves the weight state
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tensorflow.train.Checkpoint(gan)
  checkpoint.restore(tensorflow.train.latest_checkpoint(checkpoint_dir))

  # generate seed for image generation and save first image
  seed = tensorflow.random.normal([1, latent_dim])
  save_posters(generator(seed, training = False).numpy(), 0)

  #begin training loop
  print('beginning training...')
  for epoch in range(epochs):
    start = time.time()

    # reshuffle dataset each epoch
    dataset = dataset.shuffle(100)
    
    for batch in dataset:
      real_posters, _ = batch
      fake_posters = generator.predict(tensorflow.random.normal([batch_size, latent_dim]))
      real_labels = tensorflow.ones((batch_size, 1))
      fake_labels = tensorflow.zeros((batch_size, 1))

      real_disc_loss, _ = discriminator.train_on_batch(normalize_for_model(real_posters), real_labels)
      fake_disc_loss, _ = discriminator.train_on_batch(fake_posters, fake_labels)

      latent = tensorflow.random.normal([batch_size, latent_dim])

      gen_loss = gan.train_on_batch(latent, real_labels)

    if (epoch + 1) % save_checkpoint_epoch == 0:
      print('saving checkpoint at epoch: ', epoch,
      ', disc loss for real: ', real_disc_loss,
      ', disc loss for fake: ', fake_disc_loss,
      ', gen loss: ', gen_loss)
      checkpoint.save(file_prefix = checkpoint_prefix)
      # sum_performance(batch, discriminator, generator, latent_dim, batch_size)

    if (epoch + 1) % save_image_epoch == 0:
      save_posters(generator(seed, training = False).numpy(), epoch)
