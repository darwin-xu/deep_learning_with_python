from keras.datasets import reuters
(train_data, train_labels), (test_data,
                             test_labels) = reuters.load_data(num_words=10000)

word_index = reuters.get_word_index()

reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])


def decodeData(data, index):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in data[index]])


print(decodeData(train_data, 0))
