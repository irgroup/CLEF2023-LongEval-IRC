#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create an pyterrier index from the index name and the config in the config file.

Set /indexIindex_WT/data.properties to index.meta.data-source=fileinmem

Example:
    Run the system with the following command::

        $ python -m src.create_index --index WT
"""
import os
from argparse import ArgumentParser

import yaml  # type: ignore

from src.exp_logger import logger  # type: ignore
import pyterrier as pt  # type: ignore
import pyterrier_doc2query
from pyterrier_doc2query import Doc2Query, QueryScorer, QueryFilter
from pyterrier_dr import ElectraScorer
import pyterrier_colbert.indexing
with open("settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_index(index_name: str) -> pt.IndexFactory:
    """Index a dataset into pyterrier. The dataset must be in the TREC format.

    Args:
        index_name (str): Name of the dataset in the config file.
        documents_path (str): Path to the documents.

    Returns:
        pt.IndexFactory: The index as a pyterrier IndexFactory object.
    """
    index_location = os.path.join(BASE_PATH, config["index_dir"] + config[index_name]["index_name"])
    documents_path = os.path.join(BASE_PATH, config[index_name]["docs"])

    indexer = pt.TRECCollectionIndexer(
        index_location,
        meta={"docno": 26, "text": 100000},
        meta_tags={"text": "ELSE"},
        blocks=True,
        verbose=True,
    )

    documents = [os.path.join(documents_path, path) for path in os.listdir(documents_path)]
    index = indexer.index(documents)

    return index


def create_index_d2q(index_name: str) -> pt.IndexFactory:
    index_location = os.path.join(BASE_PATH, config["index_dir"] + config[index_name]["index_name"])
    documents_path = os.path.join(BASE_PATH, config[index_name]["docs"])

    doc2query = pyterrier_doc2query.Doc2Query(batch_size=16, append=True, num_samples=10, verbose=True)

    documents = [os.path.join(documents_path, path) for path in os.listdir(documents_path)]
    gen = pt.index.treccollection2textgen(
        documents,
        num_docs = 1500000,
        verbose=True,
        meta=["docno", "text"],
        tag_text_length= 100000,
        meta_tags={"text": "ELSE"}
        )

    indexer = pt.IterDictIndexer(
        index_location,
        verbose=True,
        meta={"docno": 26, "text": 100000},
        meta_tags={"text": "ELSE"},
        )

    pipeline = doc2query >> indexer

    index = pipeline.index(gen)

    return index

def create_index_d2q_minus2(index_name: str) -> pt.IndexFactory:
    index_location = os.path.join(BASE_PATH, config["index_dir"] + config[index_name]["index_name"] + "_d2q--")
    documents_path = os.path.join(BASE_PATH, config[index_name]["docs"])

    doc2query = Doc2Query(append=False, num_samples=20, verbose=True)
    scorer = ElectraScorer()


    documents = [os.path.join(documents_path, path) for path in os.listdir(documents_path)]
    gen = pt.index.treccollection2textgen(
        documents,
        num_docs = 1500000,
        verbose=True,
        meta=["docno", "text"],
        tag_text_length= 100000,
        meta_tags={"text": "ELSE"}
        )

    indexer = pt.IterDictIndexer(
        index_location,
        verbose=True,
        meta={"docno": 26, "text": 100000},
        meta_tags={"text": "ELSE"},
        )


    pipeline = doc2query >> QueryScorer(scorer) >> QueryFilter(t=3.21484375) >> indexer # t=3.21484375 is the 70th percentile for generated queries on MS MARCO

    index = pipeline.index(gen)

    return index


def create_index_colBERT(index_name: str) -> pt.IndexFactory:
    index_location = os.path.join(BASE_PATH, config["index_dir"] + config[index_name]["index_name"][:-1] + "_colBERT")
    documents_path = os.path.join(BASE_PATH, config[index_name]["docs"])

    # checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"

    documents = [os.path.join(documents_path, path) for path in os.listdir(documents_path)]
    gen = pt.index.treccollection2textgen(
        documents,
        num_docs = 1500000,
        verbose=True,
        meta=["docno", "text"],
        tag_text_length= 100000,
        meta_tags={"text": "ELSE"}
        )

    indexer = pt.IterDictIndexer(
        index_location,
        verbose=True,
        meta={"docno": 26, "text": 100000},
        meta_tags={"text": "ELSE"},
        )



    indexer = pyterrier_colbert.indexing.ColBERTIndexer(
        "data/models/colbert.dnn",
        index_location,
        "WT_colbert",
        chunksize=3,
        num_docs=1500000,
        )

    documents = [os.path.join(documents_path, path) for path in os.listdir(documents_path)]
    index = indexer.index(gen)

    return index


def main(args):
    if args.d2q:
        create_index_d2q(args.index)
    elif args.d2q_minus2:
        create_index_d2q_minus2(args.index)
    elif args.colBERT:
        create_index_colBERT(args.index)
    else:
        create_index(args.index)

if __name__ == "__main__":
    parser = ArgumentParser(description="Create an pyterrier index from a config.")

    # input arguments
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Name of the dataset in the config file",
    )

    parser.add_argument(
        "--d2q",
        required=False,
        action = 'store_true',
        help="Whether to create a doc2query index",
    )
    
    parser.add_argument(
        "--d2q_minus2",
        required=False,
        action = 'store_true',
        help="Whether to create a doc2query-- index",
    )

    parser.add_argument(
        "--colBERT",
        required=False,
        action = 'store_true',
        help="Whether to create a colBERT index",
    )

    args = parser.parse_args()

    main(args)
