import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense

from core.module_prototype import MachineLearning, fetch_mnist_data


def _build_layers(params, input_shape):
    model = Sequential()

    model = add_conv_layers(model, params, input_shape)

    # Flatten the output of convolutional layers before dense layers
    if 'conv_layers' in params:
        model.add(Flatten())

    # Add the input layer (only if no convolutional layers)
    if 'conv_layers' not in params:
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


def add_conv_layers(model, params, input_shape):
    # Add convolutional layers if specified
    if 'conv_layers' in params:
        for conv_params in params['conv_layers']:
            conv2D = Conv2D(
                conv_params['filters'],
                conv_params['kernel_size'],
                activation=conv_params['activation'],
                padding=conv_params['padding'],
                input_shape=input_shape,
                strides=conv_params.get('strides', (1, 1))
            )
            model.add(conv2D)
            model.add(MaxPooling2D((2, 2)))  # Add pooling for dimensionality reduction
    return model


def preprocess_input(X):
    """
    Preprocesses the input data by reshaping it to 28x28 image format and normalizing pixel values.

    :param X: Input data.
    :return: Preprocessed input data.
    """
    X = X.values if isinstance(X, pd.DataFrame) else X  # Convert DataFrame to NumPy array if needed
    X = X.reshape(-1, 28, 28, 1) / 255.0  # Reshape and normalize
    return X


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

        # Reshape the input data to 28x28 image format and normalize
        X_train = preprocess_input(X_train)
        X_test = preprocess_input(X_test)

        # Build the neural network model
        input_shape = (28, 28, 1)
        model = _build_layers(params, input_shape)

        # Compile the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=20, batch_size=32)

        # Evaluate the model
        accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

        # Save the trained model and evaluation results in instance attributes
        self.model = model
        self.accuracy = accuracy
        print(f"Accuracy: {accuracy}")
