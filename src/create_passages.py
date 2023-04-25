#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create passages from the LongEval dataset. For each document the top 1000 passages, 
based on BM25, are recalled. The passages are created in the MSMARCO format as triplets of query id, 
relevant passage and not relevant passage. 

Example:
    Run the system with the following command::

        $ python -m src.create_passages --index WT
"""
import json
import os
from argparse import ArgumentParser
from typing import Any, Dict, Union

from src.exp_logger import logger  # type: ignore
import pandas as pd  # type: ignore
import pyterrier as pt  # type: ignore

from tqdm import tqdm  # type: ignore

from src.load_index import setup_system  # type: ignore


def create_passeges(index_name: str) -> None:
    """Create passages for a given document index and save them to disk.

    Args:
        index_name (str): Name of the index referenced in the `config` file.
    """
    # create dirs if not exists
    if not os.path.exists("data/passages"):
        os.makedirs("data/passages")
    if not os.path.exists("data/passages/triplets"):
        os.makedirs("data/passages/triplets")
    if not os.path.exists("data/passages/collection"):
        os.makedirs("data/passages/collection")

    index, topics, qrels = setup_system(index_name)

    # Passage creation pipeline
    pipe = (
        pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"])
        >> pt.text.sliding(length=128, stride=64, prepend_attr=None, text_attr="text") % 1000
        >> pt.text.scorer(body_attr="text", wmodel="BM25")
    )

    # itterate over topics
    for topic in tqdm(topics.iterrows(), total=len(topics)):
        done = os.listdir("data/passages/triplets")
        if f"{topic[1]['qid']}.csv" in done:
            logger.info(f"Skipped {topic[1]['qid']}")
            continue
        qid = topic[1]["qid"]

        # get top 1000 passages for one query by BM25
        passages = pipe.transform(topics[topics["qid"] == qid])

        # Add Docno to passages for merging with qrels
        passages["docno_full"] = passages["docno"].str.split("%").str[0]

        # Merge qrels to passeges
        passages_graded = passages.merge(qrels, left_on="docno_full", right_on="docno").sort_values(
            "score", ascending=False
        )

        # Create triplets
        passages_rel = passages_graded[passages_graded["label"] == 1][["docno_x", "text"]]
        passages_not_rel = passages_graded[passages_graded["label"] == 0][["docno_x", "text"]]

        min_len = min([len(passages_rel), len(passages_not_rel)])

        id_list = [qid for _ in range(min_len)]

        triplets = pd.DataFrame(
            data={
                "qid": id_list,
                "pid+": passages_rel["docno_x"].to_list()[:min_len],
                "pid-": passages_not_rel["docno_x"].to_list()[:min_len],
            }
        )
        triplets.to_csv(f"data/passages/triplets/{qid}.csv", index=False, sep="\t", header=False)

        # Create collection
        passages[["docno", "text", "docno_full"]].to_csv(
            f"data/passages/collection/{qid}.csv", index=False, sep="\t", header=False
        )


def merge_passages() -> None:
    """Merge all passages created by `create_passages` and save them in the MS MARCO format."""
    basedir = "data/passages/"
    _, topics, _ = setup_system("t1")

    # merge Triplets
    with open("data/passages/triplets.train.tsv", "w") as f:
        files = os.listdir(basedir + "triplets")

        for file in files:
            with open(basedir + "triplets/" + file, "r") as f1:
                for line in f1:
                    f.write(line)
    logger.info("Merged Triplets")

    # merge collection
    with open("data/passages/collection_pre.tsv", "w") as f:
        files = os.listdir(basedir + "collection")
        for file in files:
            with open(basedir + "collection/" + file, "r") as f1:
                for line in f1:
                    f.write(line)
    logger.info("Merged Collection")

    # Drop duplicates and convert IDs to increasing integers
    # Queries
    queries = topics.reset_index()
    queries_patch = queries[["index", "qid"]]
    queries_patch.to_csv(basedir + "queries-patch.tsv", index=False, sep="\t", header=False)
    queries[["index", "query"]].to_csv(basedir + "queries.tsv", index=False, sep="\t", header=False)
    logger.info("Saved Queries")

    # Docs
    passages = (
        pd.read_csv(basedir + "collection_pre.tsv", sep="\t", header=None)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    passages = passages.reset_index()
    logger.info("Loaded Collection")

    # save docs patch
    docs_patch = passages[["index", 0, 2]]
    docs_patch.to_csv(basedir + "docs_patch.tsv", sep="\t", header=False, index=False)

    passages[["index", 1]].to_csv(basedir + "collection.tsv", sep="\t", header=False, index=False)
    logger.info("Saved Docs")

    # Triplets
    triplets = pd.read_csv("data/passages/triplets.train.tsv", sep="\t", header=None)
    triplets = triplets.merge(docs_patch, left_on=1, right_on=0).rename(columns={"index": "pid+"})
    triplets = triplets.merge(docs_patch, left_on="2_x", right_on=0).rename(
        columns={"index": "pid-"}
    )
    triplets = triplets.merge(queries_patch, left_on="0_x", right_on="qid")
    triplets = triplets[["index", "pid+", "pid-"]]

    with open("data/passages/triplets.train.tsv", "w") as f:
        for line in triplets.values.tolist():
            json.dump(line, f)
            f.write("\n")
    logger.info("Saved Triplets")


if __name__ == "__main__":
    parser = ArgumentParser(description="Create passages from a given index")

    # input arguments
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Name of the dataset in the config file",
    )

    args = parser.parse_args()

    create_passeges(args.index)
    merge_passages()
