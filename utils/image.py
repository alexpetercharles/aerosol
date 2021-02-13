from matplotlib import pyplot
import numpy

def plot_posters(posters):
  for i in range(len(posters) - 1):
      # define subplot
      pyplot.subplot(4, 5, 1 + i)
      # turn off axis
      pyplot.axis('off')
      # plot raw pixel data
      pyplot.imshow(posters[i].numpy().astype('uint8'))
  pyplot.show()

# normalize 0 : 255 to -1 : 1
def normalize(dataset):
  return (dataset - 127.5) / 127.5