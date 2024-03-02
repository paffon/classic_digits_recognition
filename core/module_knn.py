import os
import time

import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import ui.ui as ui
from core.module_prototype import MachineLearning, fetch_mnist_data


class KNN(MachineLearning):
    def __init__(self):
        super().__init__(name="K-Nearest Neighbors", input_shape=(28, 28))

    def train(self, params):
        mnist = fetch_mnist_data()  # Fetch the MNIST dataset from OpenML

        # Extract the features (X) and labels (y)
        X, y = mnist['data'], mnist['target']

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the KNN classifier with the provided parameters
        knn = KNeighborsClassifier(n_neighbors=params['k_neighbors'], weights=params['weights'])

        # Train the KNN classifier
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
