import os
from argparse import ArgumentParser

import pyterrier as pt  # type: ignore

from config import CONFIG, INDEX_DIR


def create_index(index_name: str) -> pt.IndexFactory:
    """Index a dataset into pyterrier. The dataset must be in the TREC format.

    Args:
        index_name (str): Name of the dataset in the config file.
        documents_path (str): Path to the documents.

    Returns:
        pt.IndexFactory: _description_
    """
    index_location = INDEX_DIR + CONFIG[index_name]["index_name"]
    documents_path = CONFIG[index_name]["docs"]

    indexer = pt.TRECCollectionIndexer(
        index_location,
        meta={"docno": 26, "text": 100000},
        meta_tags={"text": "ELSE"},
        blocks=True,
        verbose=True,
    )

    documents = [
        os.path.join(documents_path, path) for path in os.listdir(documents_path)
    ]
    index = indexer.index(documents)

    return index


if __name__ == "__main__":
    parser = ArgumentParser(description="Create an pyterrier index from a config.")

    # input arguments
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Name of the dataset in the config file",
    )

    args = parser.parse_args()

    create_index(args.index)
