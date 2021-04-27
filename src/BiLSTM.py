import warnings
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from time import time

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

class BiLSTM_model:
    def __init__(self):
        self.model = None
        self.history = None

    def build_model(self, X, MAX_NB_WORDS, EMBEDDING_DIM, LSTM_OUT, optimizer):
        print('>> Building model...')
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.4)) # start at 0
        model.add(Bidirectional(LSTM(LSTM_OUT))) # bidirectional layer
        #model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', 
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')])
        print('>> Compiled...')

        self.model = model
        return self.model

    def fit(self, X_train, y_train, batch_size=64, epochs=4, validation_split=0.20):
        start_time = time()
        print('>> Fitting model...')

        self.history = self.model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    validation_split=validation_split,
                    verbose = 1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        elapsed_time = time() - start_time
        print('>> Completed')
        print('>> Training duration (s): {0}'.format(elapsed_time))
        return self.model

    def evaluate(self, X_test, y_test):
        scores = self.model.evaluate(X_test, y_test)
        print('Test set\n  Loss: {:0.4f}\n  Accuracy: {:0.4f}\n  Precision: {:0.4}\n  Recall: {:0.4f}'.format(scores[0],scores[1], scores[2], scores[3]))
