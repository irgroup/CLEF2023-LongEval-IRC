#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BM25 System for LongEval 2023.

This is the baseline system for the LongEval 2023 task. It uses the default 
BM25 implementation of pyterrier and the default parameter.

Example:
    Run the system with the following command::

        $ python -m systems.BM25 --index WT
"""
import os
from argparse import ArgumentParser

import pyterrier as pt  # type: ignore

from src.exp_logger import logger
from src.load_index import setup_system

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

if not pt.started():
    pt.init()


def get_system(index: pt.IndexFactory) -> pt.BatchRetrieve:
    """Return the system as a pyterrier BatchRetrieve object.

    Args:
        index (pt.IndexFactory): The index to be used in the system.

    Returns:
        pt.BatchRetrieve: System as a pyterrier BatchRetrieve object.
    """
    bm25 = pt.BatchRetrieve(index, wmodel="BM25").parallel(4)
    return bm25


def main(args):
    filename = __file__.split("/")[-1]
    path = "results/TREC/IRCologne_" + filename[:-2] + args.index

    index, topics, _ = setup_system(args.index)

    system = get_system(index)
    results = system.transform(topics)

    pt.io.write_results(res=results, filename=path, format="trec")
    logger.info("Writing results to %s", path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Create an pyterrier index from a config.")
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Name of the dataset in the config file (WT, ST or LT)",
    )

    main(parser.parse_args())
