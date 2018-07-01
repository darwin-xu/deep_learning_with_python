import pickle
import zipfile

with ZipFile('spam.zip', 'w') as myzip:
    myzip.write('eggs.txt')

# f = gzip.open('mnist.zip', 'rb')
# data = pickle.load(f, encoding='bytes')
# f.close()
# (x_train, _), (x_test, _) = data
