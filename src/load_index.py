import os
from typing import Tuple

import pandas as pd  # type: ignore
from src.exp_logger import logger  # type: ignore

import pyterrier as pt  # type: ignore
import yaml  # type: ignore

with open("settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

TEAM = config["team"]


def load_index(index_name: str) -> pt.IndexFactory:
    """Load an index from disk.

    Args:
        index_name (str): Name of the index as specified in the config file.

    Returns:
        pt.IndexFactory: The loaded index.
    """
    index = pt.IndexFactory.of(config["index_dir"] + index_name)
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

    index = load_index(config[index_name]["index_name"])
    topics = pt.io.read_topics(config[index_name][split]["topics"])
    qrels = pt.io.read_qrels(config[index_name][split]["qrels"])
    return index, topics, qrels


def tag(system: str, index: str) -> str:
    """Create a tag for the run."""
    return f"{TEAM}-{system}.{index}"
