"""
Code for parsing online task profile into lookup table format.
"""

import json
from data_loader import DataLoader


DEFAULT_ONLINE_TASK_JSON_PATH = ("PATH/TO/NEW/TASK/DATA")


def parse_online_task_json(online_task_json_path=None):
    """Return json file for online task in lookup table format."""

    if online_task_json_path is None:
        online_task_json_path = DEFAULT_ONLINE_TASK_JSON_PATH

    with open(online_task_json_path, "r") as read_file:
        online_task = json.load(read_file)

    # Parse online task with DataLoader
    dataloader = DataLoader()
    X, Y = dataloader.process_dataset(online_task)

    # Convert online task data into lists
    list_X = [list(xi) for xi in list(X)]

    # Note: parse list_Y as errors, i.e. 1 - accuracies
    list_Y = [1 - float(yi) for yi in Y]

    # Build look up table
    tuple_X = [tuple(xi) for xi in list_X]
    look_up_table = dict(zip(tuple_X, list_Y))

    return list_X, list_Y, look_up_table


if __name__ == '__main__':
    parse_online_task_json()
