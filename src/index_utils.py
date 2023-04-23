import os
from typing import Tuple

import pandas as pd  # type: ignore
import pyterrier as pt  # type: ignore

from config import CONFIG, INDEX_DIR

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
if not pt.started():
    pt.init()


def load_index(index_name: str) -> pt.IndexFactory:
    """Load an index from disk.

    Args:
        index_name (str): Name of the index as specified in the config file.

    Returns:
        pt.IndexFactory: The loaded index.
    """
    index = pt.IndexFactory.of(INDEX_DIR + index_name)
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

    index = load_index(CONFIG[index_name]["index_name"])
    topics = pt.io.read_topics(CONFIG[index_name][split]["topics"])
    qrels = pt.io.read_qrels(CONFIG[index_name][split]["qrels"])
    return index, topics, qrels
