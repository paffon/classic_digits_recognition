import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from core.module_prototype import MachineLearning, fetch_mnist_data

import ui.ui as ui
import core.utils as my_utils


def _build_layers(params, input_shape):
    model = Sequential()

    # Add the input layer
    model.add(
        Dense(
            params['input_layer']['size'],
            activation=params['input_layer']['activation'],
            input_shape=input_shape
        )
    )
    if 'dropout' in params['input_layer']:
        model.add(Dropout(params['input_layer']['dropout']))

    # Add hidden layers
    for layer_params in params['hidden_layers']:
        model.add(
            Dense(
                layer_params['size'],
                activation=layer_params['activation'],
                kernel_initializer=layer_params['kernel_initializer']
            )
        )
        if 'dropout' in layer_params:
            model.add(Dropout(layer_params['dropout']))

    # Add the output layer
    model.add(
        Dense(
            params['output_layer']['size'],
            activation=params['output_layer']['activation']
        )
    )

    return model


class NN(MachineLearning):
    def __init__(self):
        super().__init__(name="Neural Network", input_shape=(28, 28, 1))

    def train(self, params):
        mnist = fetch_mnist_data()

        # Extract the features (X) and labels (y)
        X, y = mnist['data'], mnist['target']

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Convert labels to integers if they are not already
        y_train = y_train.astype('int')
        y_test = y_test.astype('int')

        input_shape = (X_train.shape[1],)

        # Build the neural network model
        model = _build_layers(params, input_shape)

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        model.fit(X_train, y_train, epochs=1, batch_size=32)

        # Evaluate the model
        accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

        # Save the trained model and evaluation results in instance attributes
        self.model = model
        self.accuracy = accuracy
        print(f"Accuracy: {accuracy}")