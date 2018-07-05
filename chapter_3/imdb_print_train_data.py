from keras.datasets import imdb
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()

# for key in sorted(word_index.keys()):
#     print('[' + key + ']: ', word_index[key])
# quit()

reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])

# for key in sorted(reverse_word_index.keys()):
#     print(key, ': [' + reverse_word_index[key] + ']')
# quit()


def decodeReview(index):
    return ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in train_data[index]])


def printReview(index):
    print('---', 'Positive' if train_labels[index] else 'Negative', '---')
    print(decodeReview(index))
    print()


#for i in range(100):
print(train_data[0])
printReview(0)
quit()


def encodeReview(review):
    return [word_index[word] + 3 for word in review.split()]


review = input('please input review:')

e = encodeReview(review)

print(e)