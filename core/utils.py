import json


def read_hyperparameters(model_type: str) -> dict:
    file_name = f'default_hyperparameters_{model_type}'
    full_path = f'./config/{file_name}.json'
    return read_json(full_path)


def read_json(full_path) -> dict:
    with open(full_path, 'r') as f:
        return json.load(f)
