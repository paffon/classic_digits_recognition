import ui.ui as ui
import core.module_cnn as module_cnn
import core.module_knn as module_knn
import core.utils as my_utils


def main() -> None:
    """
    Main function to train a model or load a pre-trained one, make predictions, and display results.

    :return: None
    """
    # Ask the user to choose between training live or loading a pre-trained model
    train = 'Train a brand new model. Exciting!'
    load = 'Load a pre-trained model and perform predictions!'

    train_or_load, chosen_model_name = ui.ask_user_questions(
        train, load, module_cnn.class_name, module_knn.class_name
    )

    # Determine the machine learning class to be used
    learner_class = {
        module_cnn.class_name: module_cnn.NN,
        module_knn.class_name: module_knn.KNN,
    }[chosen_model_name]

    # Get trained model (Train live or load pre-trained)
    instance = learner_class()
    if train_or_load == train:
        instance.train_and_save()
    elif train_or_load == load:
        instance.load_model()
    else:
        raise ValueError(f"Invalid action {train_or_load}")

    # Continuously predict using the trained model until interrupted
    while True:
        predict(instance)


def predict(instance) -> None:
    """
    Function to make predictions using a trained model.

    :param instance: Trained model instance
    :return: None
    """
    # Ask the user which test set they'd like to use
    chosen_test_set = ui.ask_user(
        prompt="Which test set do you want to use?",
        options=['my_samples', 'mnist_samples']
    )

    # Use the trained model of the chosen machine learning class to make predictions on the chosen test set
    predictions = instance.get_predictions_for_folder(chosen_test_set)

    total_predictions = len(predictions['correct']) + len(predictions['incorrect'])
    total_incorrect = len(predictions['incorrect'])

    # Display the results of the predictions
    print(f'{total_incorrect}/{total_predictions} incorrect:')
    for i, incorrect_prediction in enumerate(predictions['incorrect']):
        print(f'\t{i}. {incorrect_prediction}')

    # Calculate and display accuracy
    predictions_accuracy = len(predictions['correct']) / (
            len(predictions['correct']) + len(predictions['incorrect']))

    report(predictions_accuracy, instance, predictions)


def report(predictions_accuracy: float, instance, predictions) -> None:
    """
    Function to report predictions accuracy and display a heatmap of predicted vs actual labels.

    :param predictions_accuracy: Accuracy of the predictions
    :param instance: Trained model instance
    :param predictions: Predicted results
    :return: None
    """
    print(f'Test accuracy is:  {round(predictions_accuracy, 4)}\n'
          f'Model accuracy is: {round(instance.accuracy, 4)}')

    # Initialize a dictionary to store counts of predictions
    y_predicted_as_x = {i: {j: 0 for j in range(10)} for i in range(10)}

    for predictions_list in predictions.values():
        for prediction in predictions_list:
            x = prediction['predicted']
            y = prediction['actual']
            y_predicted_as_x[x][y] += 1

    # Plot a heatmap from the dictionary
    my_utils.plot_heatmap_from_dict(y_predicted_as_x)


if __name__ == "__main__":
    main()
