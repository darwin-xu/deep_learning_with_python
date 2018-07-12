import keras
import os
import scipy

# Load the images
def loadPicture(path):
    catPictures = os.listdir(path)
    imgs = []
    for pic in catPictures:
        #print(pic)
        try:
            img = scipy.misc.imread(os.path.join(path, pic))
            img = scipy.misc.imresize(img, (128, 128))
            imgs.append(img)
        except OSError:
            continue
    return imgs

imgs = loadPicture('chapter_2/cats')
print(type(imgs))

# Prepare the network
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
