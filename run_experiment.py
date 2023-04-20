import pyterrier as pt
import os
import numpy as np
import pandas as pd
from config import CONFIG, logger
from src.index_utils import setup_system
import pyterrier_colbert.ranking
import pickle
from src.LETOR import LETOR

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

if not pt.started():
    pt.init(mem=20000)


index, topics, qrels = setup_system("t1")

train_topics, validation_topics, test_topics = np.split(
    topics, [int(0.6 * len(topics)), int(0.8 * len(topics))]
)
train_qrels, validation_qrels, test_qrels = np.split(
    qrels, [int(0.6 * len(qrels)), int(0.8 * len(qrels))]
)


# Baselines
TF_IDF = pt.BatchRetrieve(index, wmodel="TF_IDF")
BM25 = pt.BatchRetrieve(index, wmodel="BM25")
PL2 = pt.BatchRetrieve(index, wmodel="PL2")


# LTR
letor = LETOR(index, query_path=CONFIG["t1"]["topics"])


def _features(row):
    docid = row["docid"]
    queryid = row["qid"]
    features = row["features"]  # get the features from WMODELs
    letor_features = letor.get_features_letor(queryid, docid)

    return np.append(features, letor_features)


lr_pipe = pickle.load(open("data/models/BM25-LR-LETOR.model", "rb"))
rf_pipe = pickle.load(open("data/models/BM25-RF-LETOR.model", "rb"))
svr_pipe = pickle.load(open("data/models/BM25-SVR-LETOR.model", "rb"))
lmart_xgb_pipe = pickle.load(open("data/models/BM25-XGB-LETOR.model", "rb"))


# ColBERT
colbert_factory = pyterrier_colbert.ranking.ColBERTFactory("data/models/colbert.dnn", None, None)
colbert = colbert_factory.text_scorer(doc_attr="text")
colbert_pipe = BM25 >> pt.text.get_text(index, "text") >> colbert


systems = [TF_IDF, BM25, PL2, rf_pipe, lr_pipe, svr_pipe, lmart_xgb_pipe, colbert_pipe]
names = [
    "TF-IDF",
    "BM25",
    "PL2",
    "Random Forest",
    "Logistic Regression",
    "Support Vector Regression",
    "LambdaMART",
    "ColBERT",
]

results = pt.Experiment(
    systems,
    test_topics,
    test_qrels,
    eval_metrics=["map", "ndcg", "P_20", "ndcg_cut_20"],
    baseline=0,
    names=names,
    correction="bonferroni",
    save_dir="./results",
    verbose=True,
)


results.to_csv("results.csv")