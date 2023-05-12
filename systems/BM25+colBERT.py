#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BM25 baseline systems reranked with colBERT.

Example:
    Create runs on the train topics of the given index::

        $ python -m systems.BM25+colBERT --index WT --train

    Create runs on the test topics of the given index::

        $ python -m systems.BM25+colBERT --index WT
"""
from argparse import ArgumentParser
from src.exp_logger import logger
import os
import pyterrier as pt  # type: ignore
from ranx import Run, fuse

from src.load_index import setup_system, tag
from src.metadata import get_metadata, write_metadata_yaml
import yaml

from pyterrier_colbert.ranking import ColBERTFactory

with open("settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

logger.setLevel("INFO")


def get_system(index) -> pt.BatchRetrieve:
    BM25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True).parallel(6)

    colbert_factory = ColBERTFactory(
        "data/models/colbert.dnn", None, None)
    colbert = colbert_factory.text_scorer(doc_attr='text')

    sparse_colbert = BM25 >> pt.text.get_text(index, 'text') >> colbert

    return sparse_colbert


def main(args):
    run_tag = tag("BM25+colBERT", args.index)
    index, topics, _ = setup_system(args.index, train=args.train)

    system = get_system(index)
    results = system.transform(topics)

    pt.io.write_results(res=results, filename=config["results_path"] + run_tag)
    write_metadata_yaml(
        config["metadata_path"] + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "bm25",
                        "method": "org.terrier.matching.models.BM25",
                        "k_1": "1.2",
                        "k_3": "8",
                        "b": "0.75",
                    },
                    "2": {
                        "name": "colBERT",
                        "method": "pyterrier_colbert.ranking.ColBERTFactory",
                        "checkpoint": "http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip",
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