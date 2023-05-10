#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""System that converts the documents into passages and uses BM25 to rank them.

Example:
    Run the system with the following command::

        $ python -m systems.BM25_passage --index WT
"""
from argparse import ArgumentParser
from src.exp_logger import logger

import pyterrier as pt  # type: ignore

from src.load_index import setup_system, tag
import yaml
import os

from pyterrier_colbert.ranking import ColBERTFactory

with open("settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def get_system() -> pt.BatchRetrieve:
    """Return the system as a pyterrier BatchRetrieve object.

    Returns:
        pt.BatchRetrieve: System as a pyterrier BatchRetrieve object.
    """
    pytcolbert = ColBERTFactory("data/models/colbert.dnn", "data/index/index_WT_colBERT", "WT_colbert", faiss_partitions=100, gpu=False)
    dense_e2e = pytcolbert.end_to_end() % 1000

    return dense_e2e


def main():
    run_tag = tag("colBERT", "colBERT_WT")

    topics = pt.io.read_topics(config["WT"]["train"]["topics"])

    system = get_system()
    results = system.transform(topics)

    pt.io.write_results(res=results, filename= config["results_path"] + run_tag)


if __name__ == "__main__":

    main()
