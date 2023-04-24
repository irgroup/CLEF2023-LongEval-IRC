#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create an pyterrier index from the index name and the config in the config file.

Example:
    Run the system with the following command::

        $ python -m src.create_index --index WT
"""
import os
from argparse import ArgumentParser

import pyterrier as pt  # type: ignore

if not pt.started():
    pt.init()

import yaml  # type: ignore

with open("settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def create_index(index_name: str) -> pt.IndexFactory:
    """Index a dataset into pyterrier. The dataset must be in the TREC format.

    Args:
        index_name (str): Name of the dataset in the config file.
        documents_path (str): Path to the documents.

    Returns:
        pt.IndexFactory: The index as a pyterrier IndexFactory object.
    """
    index_location = config["index_dir"] + config[index_name]["index_name"]
    documents_path = config[index_name]["docs"]

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
