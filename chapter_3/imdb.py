from keras.datasets import imdb
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decode_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])

import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(x_train.shape)
print(x_test.shape)

from keras import models
from keras import layers
model = models.Sequential()
layer1 = layers.Dense(16, activation='relu', input_shape=(10000, ))
model.add(layer1)
layer2 = layers.Dense(16, activation='relu')
model.add(layer2)
layer3 = layers.Dense(1, activation='sigmoid')
model.add(layer3)

# from keras.utils.vis_utils import plot_model
# #plot_model(model, to_file='imdb.png')
# print('---------------------')
# print(layer1.input_shape, layer1.output_shape)
# w = layer1.get_weights()
# print(w)
# print(len(w[0]), len(w[1]))
# print('--0-------------')
# print(w[0])
# print('--0 0-------------')
# print(len(w[0][0]))
# print('--1-------------')
# print(w[1])

# print(layer2.input_shape, layer2.output_shape)
# print(layer3.input_shape, layer3.output_shape)
#print(model.summary())
# keras.backend.shape(x)

# model.compile(
#     optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# from keras import optimizers
# model.compile(
#     optimizer=optimizers.RMSprop(lr=0.001),
#     loss='binary_crossentropy',
#     metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val))

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
#epochs = range(1, len(acc) + 1)
epochs = range(1, 20 + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
