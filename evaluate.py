#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate systems. 

If the system name is specified as in the compressed runs, the results are loaded from 
the saved run files and do not need to be computed again. 
"""
from src.exp_logger import logger  # type: ignore
from src.load_index import setup_system
import pyterrier as pt  # type: ignore
import numpy as np
from systems.LambdaMART import LETOR
import yaml

with open("settings.yml", "r") as f:
    config = yaml.safe_load(f)

from systems import BM25, BM25_RM3, LambdaMART

# from systems import ColBERT, ColBERT_LE


index, topics, qrels = setup_system("WT")

train_topics, validation_topics, test_topics = np.split(
    topics, [int(0.6 * len(topics)), int(0.8 * len(topics))]
)
train_qrels, validation_qrels, test_qrels = np.split(
    qrels, [int(0.6 * len(qrels)), int(0.8 * len(qrels))]
)
letor = LETOR(index, query_path=config["WT"]["train"]["topics"])


def _features(row):
    docid = row["docid"]
    queryid = row["qid"]
    features = row["features"]  # get the features from WMODELs
    letor_features = letor.get_features_letor(queryid, docid)
    return np.append(features, letor_features)


# systems = [BM25.get_system(index), BM25_RM3.get_system(index)]
# names = ["IRCologne_BM25.WT", "IRCologne_BM25_RM3.WT"]

systems = [BM25.get_system(index), LambdaMART.get_system(index)]
names = ["IRCologne_BM25.WT", "IRCologne_LambdaMART.WT"]

results = pt.Experiment(
    systems,
    test_topics,
    test_qrels,
    eval_metrics=["map", "ndcg", "P_20", "ndcg_cut_20"],
    baseline=0,
    names=names,
    correction="bonferroni",
    save_dir="./results/Compressed",
    verbose=True,
)

print(results)
results.to_csv("results/results-t1-.csv")
