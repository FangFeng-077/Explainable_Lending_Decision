from keras.layers import Input, Dense, Dropout
from keras.models import Model
import numpy as np


class NeuralNetwork:
    def __init__(self, n_features):
        # n_features = X_train.shape[1]
        inputs = Input(shape=(n_features,))
        dense1 = Dense(32, activation='relu')(inputs)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation="relu")(dropout2)
        dropout3 = Dropout(0.2)(dense3)
        outputs = Dense(1, activation='sigmoid')(dropout3)
        self.nn = Model(inputs=[inputs], outputs=[outputs])
        self.nn.compile(loss='binary_crossentropy', optimizer='adam')
        self.title = 'Neural Network'

    def fit(self, X_train, y_train):
        self.nn.fit(X_train.values, y_train.values, epochs=20, verbose=0)

    def predict(self, X_test):
        y_pred = self.nn.predict(X_test).ravel()

        # Converting predictions to label
        labels = (y_pred >= 0.5).astype(np.int)
        return labels

    def get_model(self):
        return self.nn

    def get_title(self):
        return self.title