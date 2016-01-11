

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from rnn_data_process import *
from keras.regularizers import l2, activity_l2


max_features = 1002
maxlen = 100 # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
maxlen, dense_len, (X_train, y_train), (X_test, y_test) = data_api(spilt_rate = 0.2, method = 1)

print('dense:', dense_len)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 67, input_length=maxlen))
model.add(GRU(67))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(dense_len, input_dim = maxlen, W_regularizer=l2(0.02), activity_regularizer=activity_l2(0.02)))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam')

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10,
          validation_data=(X_test, y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)
