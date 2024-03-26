import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense

from core.module_prototype import MachineLearning, fetch_mnist_data


def _build_layers(params, input_shape):
    """
    Build the layers of a neural network model based on the provided parameters.

    :param params: A dictionary containing parameters for building the model.
    :param input_shape: The shape of the input data.
    :return: The constructed neural network model.
    """
    model = Sequential()

    model = _add_conv_layers(model, params, input_shape)

    if 'conv_layers' in params:
        model.add(Flatten())

    if 'conv_layers' not in params:
        model = _add_input_layer(model, params['input_layer'], input_shape)

    model = _add_hidden_layers(model, params['hidden_layers'])

    model = _add_output_layer(model, params['output_layer'])

    return model


def _add_conv_layers(model, params, input_shape):
    """
    Add convolutional layers to the model if specified in the parameters.

    :param model: The neural network model.
    :param params: A dictionary containing parameters for building the model.
    :param input_shape: The shape of the input data.
    :return: The modified neural network model.
    """
    if 'conv_layers' in params:
        for layer_params in params['conv_layers']:
            model.add(
                Conv2D(
                    filters=layer_params['filters'],
                    kernel_size=layer_params['kernel_size'],
                    activation=layer_params['activation'],
                    input_shape=input_shape
                )
            )
            if 'pooling' in layer_params:
                model.add(MaxPooling2D(pool_size=layer_params['pooling']))

    return model


def _add_input_layer(model, input_layer_params, input_shape):
    """
    Add an input layer to the model.

    :param model: The neural network model.
    :param input_layer_params: Parameters for the input layer.
    :param input_shape: The shape of the input data.
    :return: The modified neural network model.
    """
    model.add(
        Dense(
            input_layer_params['size'],
            activation=input_layer_params['activation'],
            input_shape=input_shape
        )
    )
    if 'dropout' in input_layer_params:
        model.add(Dropout(input_layer_params['dropout']))

    return model


def _add_hidden_layers(model, hidden_layers_params):
    """
    Add hidden layers to the model.

    :param model: The neural network model.
    :param hidden_layers_params: Parameters for the hidden layers.
    :return: The modified neural network model.
    """
    for layer_params in hidden_layers_params:
        model.add(
            Dense(
                layer_params['size'],
                activation=layer_params['activation'],
                kernel_initializer=layer_params['kernel_initializer']
            )
        )
        if 'dropout' in layer_params:
            model.add(Dropout(layer_params['dropout']))

    return model


def _add_output_layer(model, output_layer_params):
    """
    Add an output layer to the model.

    :param model: The neural network model.
    :param output_layer_params: Parameters for the output layer.
    :return: The modified neural network model.
    """
    model.add(
        Dense(
            output_layer_params['size'],
            activation=output_layer_params['activation']
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


class_name = "Convolutional Neural Network"


class NN(MachineLearning):
    """
    Neural Network (NN) classifier implementation.
    """

    def __init__(self):
        super().__init__(name=class_name, input_shape=(28, 28, 1))

    def train(self, params: dict) -> None:
        """
        Train the NN classifier.

        :param params: Parameters for training.
        :return: None
        """
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
        model = _build_layers(params, self.input_shape)

        # Compile the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=2, batch_size=32)

        # Evaluate the model
        accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

        # Save the trained model and evaluation results in instance attributes
        self.model = model
        self.accuracy = accuracy
        print(f"Accuracy: {accuracy}")

    def predict(self, image: Image, actual: int) -> tuple[int, bool]:
        """
        Predict the class label for a given image.

        :param image: The image to predict.
        :param actual: The actual label of the image.
        :return: A tuple containing the predicted label and a boolean indicating if the prediction was correct.
        """
        print('CNN predicting')
        # Resize the image
        resized_image = image.resize(self.input_shape[:2])

        # Convert the image to grayscale
        grayscale_image = resized_image.convert('L')

        image_as_np_array = np.expand_dims(grayscale_image, axis=0)

        prediction_arr = self.model.predict(image_as_np_array)

        prediction = np.argmax(prediction_arr)

        correct_prediction = prediction == actual

        return prediction, correct_prediction
