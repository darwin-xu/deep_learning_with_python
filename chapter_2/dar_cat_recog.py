import os

import keras
import numpy as np
import scipy
from keras import layers, losses, metrics, models, optimizers
from PIL import Image


# Load the images
def loadPicture(path):
    catPictures = os.listdir(path)
    imgs = []
    for pic in catPictures:
        try:
            img = scipy.misc.imread(os.path.join(path, pic))
            img = scipy.misc.imresize(img, (64, 64))
            if img.shape[2] == 3:
                imgs.append(img)
        except OSError:
            continue
    return np.stack(imgs)


# Prepare data
imgs1 = loadPicture('chapter_2/cats')
label1 = np.full((imgs1.shape[0], ), 1.0)
label1 = np.asarray(label1).astype('float32')
imgs2 = loadPicture('chapter_2/dogs')
label2 = np.full((imgs2.shape[0], ), 0.0)
label2 = np.asarray(label2).astype('float32')
imgs = np.concatenate((imgs1, imgs2))
imgs = np.reshape(imgs, (imgs.shape[0], -1))
imgs = np.divide(imgs, 255)
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

network = models.Sequential()
network.add(layers.Dense(128, activation='relu', input_shape=(64 * 64 * 3, )))
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))
#network.add(layers.Dense(1, activation='sigmoid', input_shape=(64 * 64 * 3, )))
network.summary()

network.compile(
    optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# network.compile(
#     optimizer=optimizers.RMSprop(lr=0.001),
#     loss=losses.binary_crossentropy,
#     metrics=[metrics.binary_accuracy])

result = network.evaluate(training_sample, training_label)
print('training set, before: ', result)
result = network.evaluate(test_sample, test_label)
print('test     set, before: ', result)

# p1 = network.predict(training_sample)
# print(p1)
# p2 = network.predict(test_sample)
# print(p2)

network.fit(
    training_sample,
    training_label,
    epochs=20,
    batch_size=128,
    validation_data=(test_sample, test_label))

result = network.evaluate(training_sample, training_label)
print('training set, after : ', result)
result = network.evaluate(test_sample, test_label)
print('test     set, after : ', result)

p1 = network.predict(training_sample)
print(p1)
p2 = network.predict(test_sample)
print(p2)
