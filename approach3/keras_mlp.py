

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
#from getdata import *
import cPickle
import numpy as np

max_features = 56058
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 64

print('Loading data...')
#(X_train, y_train), (X_test, y_test) = data_api(spilt_rate = 0.2)
f = open('data.pkl', 'r')
((X_train, y_train), (X_test, y_test)) = cPickle.load(f)
f.close()
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
#model.add(Embedding(max_features, 67, input_length=maxlen))
#model.add(LSTM(256))  # try using a GRU instead, for fun
model.add(Dense(1024, input_shape=(1004 * 3,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(124))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam')

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=100,
          validation_data=(X_test, y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)
