import keras
import os
import scipy
import numpy as np


# Load the images
def loadPicture(path):
    catPictures = os.listdir(path)
    imgs = []
    for pic in catPictures:
        try:
            img = scipy.misc.imread(os.path.join(path, pic))
            img = scipy.misc.imresize(img, (128, 128))
            if img.shape[2] == 3:
                imgs.append(img)
        except OSError:
            continue
    return np.stack(imgs)


# Prepare data
imgs1 = loadPicture('chapter_2/cats')
label1 = np.full((imgs1.shape[0], ), 1.0)
imgs2 = loadPicture('chapter_2/dogs')
label2 = np.full((imgs2.shape[0], ), 0.0)
imgs = np.concatenate((imgs1, imgs2))
imgs = np.reshape(imgs, (imgs.shape[0], -1))
label = np.concatenate((label1, label2))

# Shuffle the data
si = np.arange(imgs.shape[0])
np.random.shuffle(si)
imgs = imgs[si, ]
label = label[si]

# Split into training and test set
pivot = imgs.shape[0] * 0.85
training_sample = imgs[:int(pivot), ]
training_label = label[:int(pivot)]
test_sample = imgs[int(pivot):, ]
test_label = label[int(pivot):]

# Prepare the network
from keras import models
from keras import layers

network = models.Sequential()
network.add(
    layers.Dense(512, activation='relu', input_shape=(128 * 128 * 3, )))
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(
    optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

network.fit(
    training_sample,
    training_label,
    epochs=20,
    batch_size=128,
    validation_data=(test_sample, test_label))

result = network.evaluate(test_sample, test_label)
print(result)
