from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_images.ndim)
digit = train_images[4]

# import matplotlib.pyplot as plt
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(10, activation='softmax'))
network.compile(
    optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# json = network.to_json()
# print(json)
print(network.summary())

from keras.utils.vis_utils import plot_model
#from keras.utils import plot_model
plot_model(network, to_file='model.png')

train_images = train_images.reshape((60000, 28 * 28))
print(train_images.shape)
print(train_images.dtype)
train_images = train_images.astype('float32') / 255
print(train_images.dtype)

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss, test_acc:', test_loss, test_acc)