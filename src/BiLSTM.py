import warnings
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

class BiLSTM_model:
    def __init__(self):
        self.model = None
        self.history = []

    def build_model(self, X, MAX_NB_WORDS, EMBEDDING_DIM, LSTM_OUT, optimizer):
        print('>> Building model...')
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.4)) # start at 0
        model.add(Bidirectional(LSTM(LSTM_OUT))) # bidirectional layer
        #model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=optimizer, 
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

    def plot_cm(self, X_test, y_test):
        y_pred = self.model.predict_classes(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print('Accuracy: {:0.4f}\nPrecision: {:0.4}\nRecall: {:0.4f}'.format(accuracy, precision, recall))

        cm = confusion_matrix(y_test, y_pred)

        # plot confusion matrix
        plt.style.use('default')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['funny','not funny'])
        disp.plot(cmap='Spectral_r', values_format='')
        plt.title('Confusion Matrix of Funny vs Not Funny')
        plt.tight_layout()
        #plt.savefig('confusion_matrix.png')
        plt.show()

    def plot_recall(self):
        plt.title('Recall')
        plt.plot(self.history.history['recall'], label='train')
        plt.plot(self.history.history['val_recall'], label='test')
        plt.ylabel('recall')
        plt.xlabel('epoch')
        plt.legend()
       #plt.savefig('model_recall.png')
        plt.show()