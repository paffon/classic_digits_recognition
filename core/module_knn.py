import os
import time

import joblib
import numpy as np
import PIL.Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import ui.ui as ui
import core.utils as my_utils
from core.module_prototype import MachineLearning, fetch_mnist_data


class_name = "K-Nearest Neighbors"


class KNN(MachineLearning):
    """
    K Nearest Neighbors (KNN) classifier implementation.
    """

    def __init__(self):
        super().__init__(name=class_name, input_shape=(28, 28))

    def train(self, params: dict) -> None:
        """
        Train the KNN classifier.

        :param params: Parameters for training.
        :return: None
        """
        mnist = fetch_mnist_data()  # Fetch the MNIST dataset from OpenML

        # Extract the features (X) and labels (y)
        X, y = mnist['data'], mnist['target']

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the KNN classifier with the provided parameters
        knn = KNeighborsClassifier(n_neighbors=params['k_neighbors'], weights=params['weights'])

        # Train the KNN classifier incrementally
        pieces = 10
        for i in range(1, pieces + 1):  # Simulate training progress from 1% to 100%
            print(ui.progress_bar(i, pieces))
            knn.fit(X_train[:len(X_train) * i // pieces],
                    y_train[:len(y_train) * i // pieces])  # Incremental training

        # Evaluate the model
        accuracy = knn.score(X_test, y_test)

        # Save the trained model and evaluation results in instance attributes
        self.model = knn
        self.accuracy = accuracy

    def predict(self, image: PIL.Image, actual: int) -> tuple[int, bool]:
        """
        Predict the class label for a given image.

        :param image: The image to predict.
        :param actual: The actual label of the image.
        :return: A tuple containing the predicted label and a boolean indicating if the prediction was correct.
        """
        print('KNN predicting')

        # Resize the image
        resized_image = image.resize((28, 28))

        # Convert the image to grayscale
        grayscale_image = resized_image.convert('L')

        # Image as 2D array
        image_as_2d = np.array(grayscale_image)

        # Flatten the image
        flattened_image = image_as_2d.flatten()

        # Reshape to ensure it's a 2D array
        flattened_image = flattened_image.reshape(1, -1)

        # Convert to a dataframe matching the training set features names
        as_df = my_utils.numpy_array_to_dataframe(flattened_image)

        # Use the trained KNN model for prediction
        prediction_arr = self.model.predict(as_df)

        prediction = int(prediction_arr[0])

        correct_prediction = prediction == actual

        return prediction, correct_prediction
