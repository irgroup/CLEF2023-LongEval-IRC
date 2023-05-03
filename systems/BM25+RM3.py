#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BM25 baseline systems including pseudo relevance feedback and rank fusion.

Example:
    Run the system with the following command::

        $ python -m systems.BM25 --index WT
"""
from argparse import ArgumentParser
from src.exp_logger import logger

import pyterrier as pt  # type: ignore
from ranx import Run, fuse

from src.load_index import setup_system, tag
from src.metadata import get_metadata, write_metadata_yaml

logger.setLevel("INFO")


results_path = "results/trec/"
metadata_path = "results/metadata/"


def main(args):
    index, topics, _ = setup_system(args.index)

    # BM25
    run_tag = tag("BM25", args.index)
    BM25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True).parallel(6)
    pt.io.write_results(BM25(topics), results_path + run_tag)
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
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
                    }
                },
            },
        },
    )

    # Pseudo relevance feedback
    # BM25 + RM3
    run_tag = tag("BM25+RM3", args.index)
    rm3_pipe = BM25 >> pt.rewrite.RM3(index) >> BM25
    pt.io.write_results(rm3_pipe(topics), results_path + run_tag, run_name=run_tag)
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
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
                        "name": "RM3 query expansion",
                        "method": "pyterrier.rewrite.RM3",
                        "fb_terms": "10",
                        "fb_docs": "3",
                        "fb_lambda": "0.6",
                        "reranks": "bm25",
                    },
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

    main(parser.parse_args())
