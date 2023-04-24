#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ColBERT LongEval System for LongEval 2023.

This system uses BM25 as a first stage ranker and the PyTerrier ColBERT model as a second stage 
re-ranker. The Model was originally trained on the MS MARCO. For further information please refer
to the GitHub repository of the PyTerrier ColBERT model (https://github.com/terrierteam/pyterrier_colbert). 
Or the original paper of the ColBERT model (https://arxiv.org/abs/2004.12832).

TODO: For training the model please refere to the other script.

Example:
    Run the system with the following command::

        $ python -m systems.ColBERT --index WT
"""
import os
from argparse import ArgumentParser

import pyterrier as pt  # type: ignore

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
if not pt.started():
    pt.init()

import pyterrier_colbert.ranking  # type: ignore

from src.exp_logger import logger
from src.load_index import setup_system


def get_system(index: pt.IndexFactory) -> pt.BatchRetrieve:
    """Return the system as a pyterrier BatchRetrieve object.
    The ColBERT checkpoint can be downloaded from "http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
    and should be placed in the "data/models" folder. For further information
    please refer to "https://trec.nist.gov/pubs/trec29/papers/uogTr.DL.pdf".

    Args:
        index (pt.IndexFactory): The index to be used in the system.

    Returns:
        pt.BatchRetrieve: System as a pyterrier BatchRetrieve object.
    """
    BM25 = pt.BatchRetrieve(index, wmodel="BM25")

    colbert_factory = pyterrier_colbert.ranking.ColBERTFactory(
        "data/models/colbert.dnn", None, None
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
