import os
from typing import Tuple

import pandas as pd  # type: ignore
import numpy as np
from src.exp_logger import logger  # type: ignore

import pyterrier as pt  # type: ignore
import yaml  # type: ignore

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_PATH, "settings.yml"), "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def _load_index(index_name: str) -> pt.IndexFactory:
    """Load an index from disk.

    Args:
        index_name (str): Name of the index as specified in the config file.

    Returns:
        pt.IndexFactory: The loaded index.
    """
    index = pt.IndexFactory.of(os.path.join(BASE_PATH, config["index_dir"] + index_name))
    print(
        "Loaded index with ",
        index.getCollectionStatistics().getNumberOfDocuments(),
        "documents.",
    )
    return index


def setup_system(
    index_name: str, train: bool = True
) -> Tuple[pt.IndexFactory, pd.DataFrame, pd.DataFrame]:
    """Load the index, topics and qrels for a dataset that is allready indexed.

    Args:
        index_name (str): Name of the dataset split as specified in the config file.
        train (bool, optional): Return the train or the test split. Defaults to True.

    Returns:
        (pt.IndexFactory, pd.DataFrame, pd.DataFrame): The index, topics and qrels.
    """
    if train:
        split = "train"
    else:
        split = "test"

    index = _load_index(config[index_name]["index_name"])
    topics = pt.io.read_topics(os.path.join(BASE_PATH, config[index_name][split]["topics"]))
    if train:
        qrels = pt.io.read_qrels(os.path.join(BASE_PATH, config[index_name][split]["qrels"]))
    else:
        qrels = None

    return index, topics, qrels


def tag(system: str, index: str) -> str:
    """Create a tag for the run."""
    team = config["team"]
    return f"{team}-{system}.{index}"


def get_train_splits(topics, qrels):
    def filter_ids(topics):
        needed_ids = list(topics["qid"].unique())  # needed ids
        qrels_split = qrels[qrels["qid"].isin(needed_ids)]
        diff = len(needed_ids) - len(qrels_split["qid"].unique())
        return qrels_split

    # split topics
    train_topics, validation_topics, test_topics = np.split(
        topics, [int(0.6 * len(topics)), int(0.8 * len(topics))]
    )

    # split qrels
    train_qrels = filter_ids(train_topics)
    validation_qrels = filter_ids(validation_topics)
    test_qrels = filter_ids(test_topics)

    return train_topics, validation_topics, test_topics, train_qrels, validation_qrels, test_qrels
