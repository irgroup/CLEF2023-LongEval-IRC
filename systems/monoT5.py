#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MonoT5 baseline system for LongEval 2023.

This system uses the monoT5 reranker and a BM25 first stage ranker.

Example:
    Run the system with the following command::

        $ python -m systems.monoT5 --index WT
"""
from argparse import ArgumentParser
from src.exp_logger import logger

import pyterrier as pt  # type: ignore
from pyterrier_t5 import MonoT5ReRanker

from src.load_index import setup_system

logger.setLevel("INFO")


def get_system(index: pt.IndexFactory) -> pt.BatchRetrieve:
    monoT5 = MonoT5ReRanker(verbose=True, batch_size=8)
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True).parallel(6)

    mono_pipeline = bm25 >> pt.text.get_text(index, "text") >> monoT5

    return mono_pipeline


def main(args):
    index, topics, _ = setup_system(args.index)

    monoT5 = get_system(index)

    pt.io.write_results(monoT5(topics), "results/trec/IRCologne-monoT5." + args.index)


if __name__ == "__main__":
    parser = ArgumentParser(description="Create an pyterrier index from a config.")
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Name of the dataset in the config file (WT, ST or LT)",
    )

    main(parser.parse_args())
