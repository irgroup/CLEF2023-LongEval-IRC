#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BM25 system run on an index expanded with doc2query.

Example:
    Create runs on the train topics of the given index::

        $ python -m systems.d2q+BM25 --index WT_d2q --train
    
    Create runs on the test topics of the given index::

        $ python -m systems.d2q+BM25 --index WT_d2q
"""
from argparse import ArgumentParser
from src.exp_logger import logger

import pyterrier as pt  # type: ignore
from src.load_index import setup_system, tag
from src.metadata import write_metadata_yaml
import yaml
logger.setLevel("INFO")


with open("settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

def get_system(index) -> pt.BatchRetrieve:
    BM25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True).parallel(6)
    return BM25


def main(args):
    index, topics, _ = setup_system(args.index, train=args.train)

    # BM25
    run_tag = tag("d2q+BM25", args.index[:2])
    system = get_system(index_path=index, index=args.index)
    pt.io.write_results(system(topics), config["results_path"] + run_tag)
    write_metadata_yaml(
        config["metadata_path"] + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "indexing": {
                    "Doc2Query": {
                        "method": "pyterrier_doc2query.Doc2Query",
                        "checkpoint": "macavaney/doc2query-t5-base-msmarco",
                        "num_samples": 10
                    }
                },
                "retrieval": {
                    "1": {
                        "name": "bm25",
                        "method": "org.terrier.matching.models.BM25",
                        "k_1": "1.2",
                        "k_3": "8",
                        "b": "0.75",
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
