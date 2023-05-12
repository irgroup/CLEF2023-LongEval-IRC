#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ColBER dense retrieval system.

Example:
    Create runs on the train topics of the given index::

        $ python -m systems.colBERT --index WT_colBERT --train

    Create runs on the test topics of the given index::

        $ python -m systems.colBERT --index WT_colBERT
    
"""
from argparse import ArgumentParser
from src.exp_logger import logger

import pyterrier as pt  # type: ignore
from src.metadata import get_metadata, write_metadata_yaml

from src.load_index import setup_system, tag
import yaml
import os

from pyterrier_colbert.ranking import ColBERTFactory

with open("settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def get_system(index_path, index) -> pt.BatchRetrieve:
    """Return the system as a pyterrier BatchRetrieve object.

    Returns:
        pt.BatchRetrieve: System as a pyterrier BatchRetrieve object.
    """
    pytcolbert = ColBERTFactory("data/models/colbert.dnn", index_path, index, faiss_partitions=100, gpu=False)
    dense_e2e = pytcolbert.end_to_end() % 1000

    return dense_e2e


def main(args):
    run_tag = tag(args.index[3:], args.index[2:])

    slice = "train" if args.train else "test"
    topics = pt.io.read_topics(config[args.index][slice]["topics"])
    index_path = os.path.join(config["index_dir"], args.index)

    system = get_system(index_path=index_path, index=args.index)
    results = system.transform(topics)

    pt.io.write_results(res=results, filename=config["results_path"] + run_tag)

    write_metadata_yaml(
            config["metadata_path"] + run_tag + ".yml",
            {
                "tag": run_tag,
                "method": {
                    "indexing": {
                        "colBERT": {
                            "method": "pyterrier_colbert.indexing.ColBERTIndexer",
                            "checkpoint": "http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip",
                        }
                    },
                    "retrieval": {
                        "1": {
                            "name": "colBERT",
                            "method": "pyterrier_colbert.ranking.ColBERTFactory.end_to_end",
                        }
                    },
                },
            },
        )

if __name__ == "__main__":
    parser = ArgumentParser(description="Create an pyterrier index from a config.")
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Name of the dataset in the config file (WT, ST or LT)",
    )
    parser.add_argument(
        "--train",
        required=False,
        action="store_true",
        help="Use the train topics to create the.",
    )

    args = parser.parse_args()
    main(args)
