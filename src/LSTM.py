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
from tensorflow.keras.layers import Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

class LSTM_model:
    def __init__(self):
        self.model = None
        self.history = []

    def tokenize(self, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, df):
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
        tokenizer.fit_on_texts(df.text.values)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        X = tokenizer.texts_to_sequences(df.text.values)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', X.shape)

        y = df['funny or not'].values
        print('Shape of label tensor:', y.shape)

        return X, y

    def split_train_test(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

        print('Shape of X and y train:', X_train.shape, y_train.shape)
        print('Shape of X and y test:', X_test.shape, y_test.shape)

        return X_train, X_test, y_train, y_test

    def balance_train_undersample(self, X_train, y_train):
        X_train_funny = pd.DataFrame(X_train[y_train == 1])
        X_train_not_funny = pd.DataFrame(X_train[y_train == 0]).sample(n=len(X_train_funny), random_state=42)

        y_train_funny = pd.DataFrame(y_train[y_train == 1])
        y_train_not_funny = pd.DataFrame(y_train[y_train == 0]).sample(n=len(y_train_funny), random_state=42)

        X_train_balanced = pd.concat([X_train_funny, X_train_not_funny])
        y_train_balanced = pd.concat([y_train_funny, y_train_not_funny])

        print('Shape of balanced X and y train:', X_train_balanced.shape, y_train_balanced.shape)

        return X_train_balanced, y_train_balanced

    def build_model1(self, X, MAX_NB_WORDS, EMBEDDING_DIM, LSTM_OUT, optimizer):
        print('>> Building model...')
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.4)) 
        model.add(LSTM(LSTM_OUT, dropout=0.2, recurrent_dropout=0.2)) 
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=optimizer, 
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')])
        print('>> Compiled')

        self.model = model
        return self.model

    def build_model2(self, X, MAX_NB_WORDS, EMBEDDING_DIM, LSTM_OUT_1, LSTM_OUT_2, optimizer):
        print('>> Building model...')
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.2)) 
        model.add(LSTM(LSTM_OUT_1, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(LSTM_OUT_2, dropout=0.2, recurrent_dropout=0.2)) 
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', 
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')])
        print('>> Compiled')

        self.model = model
        return self.model

    def fit(self, X_train, y_train, batch_size=64, epochs=3, validation_split=0.20):
        start_time = time()
        print('>> Fitting model...')

        self.history = self.model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    validation_split=validation_split,
                    verbose = 1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        print(self.model.summary())

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