#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MonoT5 system, finetuned on the WT slice of the dataset for LongEval 2023.

This system is trained on the WT dataset train qrels.

Example:
    Run the system with the following command::

        $ python -m systems.monoT5_WT --index WT
    
    If only the model trained on the dev train slice should be used::

        $ python -m systems.monoT5_WT --index WT --train
"""
from argparse import ArgumentParser
from src.exp_logger import logger

import pyterrier as pt  # type: ignore
from pyterrier_t5 import MonoT5ReRanker
import numpy as np

from src.load_index import setup_system

logger.setLevel("INFO")


def get_system(index: pt.IndexFactory, train: bool) -> pt.BatchRetrieve:
    path = f"data/models/monoT5-WT"
    if train:
        path = path + "-train"

    monoT5 = MonoT5ReRanker(verbose=True, batch_size=8, model=path)
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True).parallel(6)

    mono_pipeline = bm25 >> pt.text.get_text(index, "text") >> monoT5

    return mono_pipeline


def main(args):
    index, topics, _ = setup_system(args.index)

    monoT5 = get_system(index, args.train)

    if args.train:
        path = "results/trec/IRCologne-monoT5_WT-train."
    else:
        path = "results/trec/IRCologne-monoT5_WT."

    pt.io.write_results(monoT5(topics), path + args.index)


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
        help="Use the model trained on the train split only.",
    )

    main(parser.parse_args())
