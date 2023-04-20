import json
import os
import pickle

import numpy as np
import pandas as pd
import pyterrier as pt
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from config import CONFIG, logger
from src.index_utils import setup_system
from src.LETOR import LETOR

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"


if not pt.started():
    pt.init()


index, topics, qrels = setup_system("t1")

train_topics, validation_topics, test_topics = np.split(
    topics, [int(0.6 * len(topics)), int(0.8 * len(topics))]
)
train_qrels, validation_qrels, test_qrels = np.split(
    qrels, [int(0.6 * len(qrels)), int(0.8 * len(qrels))]
)


letor = LETOR(index, query_path=CONFIG["t1"]["topics"])


def _features(row):
    docid = row["docid"]
    queryid = row["qid"]
    features = row["features"]  # get the features from WMODELs
    letor_features = letor.get_features_letor(queryid, docid)

    return np.append(features, letor_features)


fbr = pt.FeaturesBatchRetrieve(
    index,
    controls={"wmodel": "BM25"},
    features=[
        "WMODEL:Tf",
        "WMODEL:TF_IDF",
        "WMODEL:BM25",
    ],
) >> pt.apply.doc_features(_features)


# Create the regressor object.
logger.info("Start Random Forrest training")
rf = RandomForestRegressor(
    n_estimators=10, max_depth=2, n_jobs=12, random_state=42, verbose=3, max_samples=100
)
rf_pipe = fbr >> pt.ltr.apply_learned_model(rf)
rf_pipe.fit(train_topics, train_qrels)

pickle.dump(rf_pipe, open("./data/models/BM25-RF-LETOR.model", "wb"))
logger.info("Random Forest done")


# Logistic regression
logger.info("Start Logistic regression training")
lr = LogisticRegression(random_state=42, verbose=3)
lr_pipe = fbr >> pt.ltr.apply_learned_model(lr)
lr_pipe.fit(train_topics, train_qrels, validation_topics, validation_qrels)

pickle.dump(lr_pipe, open("./data/models/BM25-LR-LETOR.model", "wb"))
logger.info("Logistic Regression done")


# Support Vector regression
logger.info("Start Support Vector Regression training")
svr = svm.SVR(verbose=3)
svr_pipe = fbr >> pt.ltr.apply_learned_model(svr)
svr_pipe.fit(train_topics, train_qrels, validation_topics, validation_qrels)

pickle.dump(svr_pipe, open("./data/models/BM25-SVR-LETOR.model", "wb"))
logger.info("Support Vector Regression done")


# LambdaMART
logger.info("Start XGBoost training")
lmart_x = xgb.sklearn.XGBRanker(
    objective="rank:ndcg",
    learning_rate=0.1,
    gamma=1.0,
    min_child_weight=0.1,
    max_depth=6,
    verbose=100,
    random_state=42,
)
lmart_xgb_pipe = fbr >> pt.ltr.apply_learned_model(lmart_x, form="ltr")
lmart_xgb_pipe.fit(train_topics, train_qrels, validation_topics, validation_qrels)
pickle.dump(lmart_xgb_pipe, open("./data/models/BM25-XGB-LETOR.model", "wb"))
logger.info("XGBoost done")
