import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_hyperparameters(model_type: str) -> dict:
    file_name = f'default_hyperparameters_{model_type}'
    full_path = f'./config/{file_name}.json'
    return read_json(full_path)


def read_json(full_path) -> dict:
    with open(full_path, 'r') as f:
        return json.load(f)


def print_2d_grayscale_image(image):
    for row in image:
        for pixel in row:
            if pixel < 1:
                pixel *= 255
            print(f'{int(pixel):4}', end='')
        print()


def numpy_array_to_dataframe(np_arr):
    # Create a DataFrame with one row
    df = pd.DataFrame(np_arr, columns=[f'pixel{i}' for i in range(1, np_arr.shape[1] + 1)])
    return df


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


def plot_heatmap_from_dict0(data: dict) -> None:
    """
    Plot a heatmap showing the frequency of each y value associated with each x value.

    :param data: Dictionary containing counts of y values associated with x values.
                 Format: data[x][y] = count
    """
    # Extracting x and y values
    x_values = sorted(data.keys())
    y_values = sorted({y for x in data.values() for y in x.keys()})

    # Creating a matrix to hold the counts
    matrix = np.zeros((len(x_values), len(y_values)))

    # Filling the matrix with counts
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            matrix[i, j] = data.get(x, {}).get(y, 0)

    # Plotting the heatmap
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Count')
    plt.xticks(np.arange(len(y_values)), y_values)
    plt.yticks(np.arange(len(x_values)), x_values)
    plt.xlabel('Y values')
    plt.ylabel('X values')
    plt.title('Heatmap of Counts')

    # Annotating data values
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            plt.text(j, i, str(int(matrix[i, j])), ha='center', va='center', color='white')

    plt.show()


def plot_heatmap_from_dict(data: dict) -> None:
    """
    Plot a heatmap showing the frequency of each y value associated with each x value.

    :param data: Dictionary containing counts of y values associated with x values.
                 Format: data[x][y] = count
    """
    x_values = list(data.keys())
    y_values = set()
    for counts in data.values():
        y_values.update(counts.keys())

    # Reverse the order of y_values to display bottom-up
    y_values = list(reversed(sorted(y_values)))

    heatmap_data = [[data[x].get(y, 0) for y in y_values] for x in x_values]

    ax = sns.heatmap(heatmap_data, xticklabels=x_values, yticklabels=y_values, cmap='viridis')

    # Add data labels to each cell
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            ax.text(j + 0.5, i + 0.5, str(heatmap_data[i][j]), ha='center', va='center', color='black')

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Amount of times that value x was classified as y')
    plt.show()

