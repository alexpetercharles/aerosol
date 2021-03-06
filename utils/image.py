from matplotlib import pyplot

generated_folder = 'data/generated/'

def save_posters(posters, epoch, n = 5):
  posters = normalize_to_float(posters)
  # for i in range(5*5):
    # define subplot
    # pyplot.subplot(n, n, 1 + i)
    # turn off axis
  pyplot.axis('off')
    # plot raw pixel data
  pyplot.imshow(posters[0].astype('float32'))
  filename = generated_folder + '/generated_poster_e%03d.png' % (epoch+1)
  pyplot.savefig(filename)
  pyplot.close()

# normalize 0 : 255 to -1 : 1
def normalize_for_model(dataset):
  return (dataset - 127.5) / 127.5

# normalize -1 : 1 to 0 : 1
def normalize_to_float(dataset):
  return (dataset + 1) / 2