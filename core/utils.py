import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def read_hyperparameters(model_type: str) -> dict:
    file_name = f'default_hyperparameters_{model_type}'
    full_path = f'./config/{file_name}.json'
    return read_json(full_path)


def read_json(full_path) -> dict:
    with open(full_path, 'r') as f:
        return json.load(f)


def save_as_png(data: pd.Series, filename: str):
    """
    Save a 28x28 image represented by a Pandas Series as a PNG file.

    :param data: Pandas Series representing the image data.
    :param filename: Name of the PNG file to save.
    """
    # Convert the Series to a NumPy array
    data_array = data.to_numpy()

    # Reshape the data into a 28x28 array
    image = data_array.reshape(28, 28)

    # Plot the image
    plt.imshow(image, cmap='gray')

    # Remove axis ticks
    plt.axis('off')

    # Save the plot as a PNG file
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    # Close the plot to free up memory
    plt.close()
