#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BM25 System for LongEval 2023.

This is the baseline system for the LongEval 2023 task. It uses the default 
BM25 implementation of pyterrier and the default parameter.

Example:
    Run the system with the following command::

        $ python -m systems.LambdaMART_USE --index WT
"""
import json
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from src.exp_logger import logger
import pyterrier as pt  # type: ignore
import spacy
import xgboost as xgb
from tqdm import tqdm

from src.load_index import setup_system

logger.setLevel("INFO")


def create_features(index: pt.IndexFactory, topics: pd.DataFrame):
    base_dir = "data/use"
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("universal_sentence_encoder")

    # queries
    query_file = "query_USE.jsonl"
    if query_file in os.listdir(base_dir):
        logger.info("Skipping USE for queries. File allready exists.")
    else:
        logger.info(f"Start calculating USE for {len(topics)} queries...")
        with open(os.path.join(base_dir, query_file), "w") as file:
            with tqdm(total=len(topics)) as pbar:
                for _, topic in topics.iterrows():
                    doc = nlp(topic["query"])
                    file.write(
                        json.dumps({"qid": topic["qid"], "use": doc.vector.tolist()})
                    )
                    file.write("\n")
                    pbar.update(1)
        logger.info("Calculating USE for queries done.")

    # docs
    logger.info("Retrieve BM25 baseline...")
    doc_file = "docs_USE.jsonl"

    cache = []
    if doc_file in os.listdir(base_dir):
        logger.info("Load cache...")
        with open(os.path.join(base_dir, doc_file), "r") as file:
            for line in file.readlines():
                cache.append(json.loads(line)["docno"])

    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    pipe = bm25 >> pt.text.get_text(index, "text")

    for topic in tqdm(topics.iterrows(), total=len(topics)):
        qid = topic[1]["qid"]
        docs = pipe.transform(topics[topics["qid"] == qid])
        docs = docs[["docno", "text"]]

        logger.info(f"Start calculating USE for {len(docs)} docs...")
        for _, text in docs.iterrows():
            if text["docno"] in cache:
                logger.info("Cache hit for document {}".format(text["docno"]))
                continue
            with open(os.path.join(base_dir, doc_file), "a+") as file:
                doc = nlp(text["text"])
                file.write(
                    json.dumps({"docno": text["docno"], "use": doc.vector.tolist()})
                )
                file.write("\n")
            cache.append(text["docno"])
    logger.info("Calculating USE for top docs done.")


def load_features(index_name: str, split: str):
    features = []
    with open(f"data/use/{split}_USE_{index_name}.jsonl") as file:
        for line in file.readlines():
            sample = json.loads(line)
            features.append(sample)
    return features


def train(
    index: pt.IndexFactory, topics: pd.DataFrame, qrels: pd.DataFrame, index_name: str
):
    logger.info("Training the model.")

    train_topics, validation_topics, test_topics = np.split(
        topics, [int(0.6 * len(topics)), int(0.8 * len(topics))]
    )
    train_qrels, validation_qrels, test_qrels = np.split(
        qrels, [int(0.6 * len(qrels)), int(0.8 * len(qrels))]
    )

    query_features = load_features(index_name, "query")
    docs_features = load_features(index_name, "docs")

    def _features(row):
        docid = row["docid"]
        queryid = row["qid"]
        features = row["features"]  # get the features from WMODELs

        # letor_features = letor.get_features_letor(queryid, docid)
        return np.append(features, query_features[queryid], docs_features[docid])

    fbr = pt.FeaturesBatchRetrieve(
        index,
        controls={"wmodel": "BM25"},
        features=[
            "WMODEL:TF_IDF",
            "WMODEL:BM25",
        ],
    ) >> pt.apply.doc_features(_features)

    lmart_x = xgb.sklearn.XGBRanker(
        objective="rank:ndcg",
        learning_rate=0.1,
        gamma=1.0,
        min_child_weight=0.1,
        max_depth=6,
        random_state=42,
        verbosity=3,
    )
    logger.info("Training LambdaMART model started...")
    LambdaMART_pipe = fbr >> pt.ltr.apply_learned_model(lmart_x, form="ltr")
    LambdaMART_pipe.fit(train_topics, train_qrels, validation_topics, validation_qrels)
    logger.info("Training LambdaMART model finished.")

    logger.info("Save model to disk...")
    pickle.dump(
        LambdaMART_pipe,
        open("./data/models/BM25-XGB-USE.model", "wb"),
    )


def get_system(index: pt.IndexFactory):
    logger.info("Loading LambdaMART model...")
    LambdaMART_pipe = pickle.load(open("data/models/BM25-XGB-USE.model", "rb"))
    return LambdaMART_pipe


def main(args):
    filename = __file__.split("/")[-1]
    path = "results/TREC/IRCologne_" + filename[:-2] + args.index

    index, topics, _ = setup_system(args.index)

    if args.features:
        create_features(index, topics)

    elif args.train:
        train(index)

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

    parser.add_argument(
        "--train",
        required=False,
        action="store_true",
        help="Train the model.",
    )

    parser.add_argument(
        "--features",
        required=False,
        action="store_true",
        help="Create features for the model.",
    )

    main(parser.parse_args())
