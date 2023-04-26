#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ColBERT LongEval System for LongEval 2023.

This system uses BM25 as a first stage ranker and a ColBERT model that was trained on the 
LongEval WT dataset as a second stage re-ranker.

TODO: For training the model please refere to the other script.

Example:
    Run the system with the following command::

        $ python -m systems.ColBERT_LE --index WT
"""
from argparse import ArgumentParser
from src.exp_logger import logger

import pyterrier as pt  # type: ignore
import pyterrier_colbert.ranking  # type: ignore

from src.load_index import setup_system


def get_system(index: pt.IndexFactory) -> pt.BatchRetrieve:
    """Return the system as a pyterrier BatchRetrieve object.

    Args:
        index (pt.IndexFactory): The index to be used in the system.

    Returns:
        pt.BatchRetrieve: System as a pyterrier BatchRetrieve object.
    """
    BM25 = pt.BatchRetrieve(index, wmodel="BM25").parallel(6)

    colbert_factory = pyterrier_colbert.ranking.ColBERTFactory(
        "data/models/colbert-t1.dnn", None, None
    )
    colbert = colbert_factory.text_scorer(doc_attr="text")
    colbert_pipe = BM25 >> pt.text.get_text(index, "text") >> colbert
    return colbert_pipe


def main(args):
    filename = __file__.split("/")[-1]
    path = "results/TREC/IRCologne_" + filename[:-2] + args.index

    index, topics, _ = setup_system(args.index)

    system = get_system(index)
    results = system.transform(topics)

    pt.io.write_results(res=results, filename=path, format="trec")
    pt.io.write_results(
        res=results,
        filename=path.replace("TREC", "Compressed") + ".res.gz",
        format="trec",
    )
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
