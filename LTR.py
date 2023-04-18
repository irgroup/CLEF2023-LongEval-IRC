
import pyterrier as pt
import os
import numpy as np
import pandas as pd

if not pt.started():
  pt.init()


## Create Index
# !rm -rf ./index_t1
# indexer = pt.TRECCollectionIndexer("data/publish/English/Documents/Trec/", blocks=True, verbose=True)


# doc_paths_t1 = [os.path.join(dataset_path_t1, path) for path in os.listdir(dataset_path_t1)]
# indexref_t1 = indexer.index(doc_paths_t1)


index_t1 = pt.IndexFactory.of("./index_t1")

query_path_t1 = "data/publish/English/Queries/train.trec"
topics_t1 = pt.io.read_topics(query_path_t1)

qrels_t1 = pt.io.read_qrels("data/publish/French/Qrels/train.txt")


train_topics, validation_topics, test_topics = np.split(
            topics_t1, [int(0.6 * len(topics_t1)), int(0.8 * len(topics_t1))]
        )
train_qrels, validation_qrels, test_qrels = np.split(
            qrels_t1, [int(0.6 * len(qrels_t1)), int(0.8 * len(qrels_t1))]
        )


print(index_t1.getCollectionStatistics().toString())


## Features
from src.LETOR import LETOR

letor = LETOR(index_t1, query_path_t1)

def _features(row):
    docid = row["docid"]
    
    queryid = row["qid"]
    features = row["features"]  # get the features from the previous stage

    letor_features = letor.get_features_letor(queryid, docid)

    return np.append(features, letor_features)


## Retrieval
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb


TF_IDF = pt.BatchRetrieve(index_t1, wmodel="TF_IDF")
BM25 = pt.BatchRetrieve(index_t1, wmodel="BM25")
PL2 = pt.BatchRetrieve(index_t1, wmodel="PL2")


fbr = pt.FeaturesBatchRetrieve(index_t1, 
                             controls = {"wmodel": "BM25"}, 
                             features=[
                                      "WMODEL:Tf",
                                      "WMODEL:TF_IDF", 
                                      "WMODEL:BM25", 
                                      ]
                             ) >> pt.apply.doc_features(_features)


# fbr.transform("Airport hello car")


# Create the regressor object.
rf = RandomForestRegressor(n_estimators=400)
rf_pipe = fbr >> pt.ltr.apply_learned_model(rf)
rf_pipe.fit(train_topics, train_qrels)

# Logistic regression
lr = LogisticRegression()
lr_pipe = fbr >> pt.ltr.apply_learned_model(lr)
lr_pipe.fit(train_topics, train_qrels)

# Support Vector regression
svr = svm.SVR()
svr_pipe = fbr >> pt.ltr.apply_learned_model(svr)
svr_pipe.fit(train_topics, train_qrels)

# LambdaMART
lmart_x = xgb.sklearn.XGBRanker(objective='rank:ndcg',
    learning_rate=0.1,
    gamma=1.0,
    min_child_weight=0.1,
    max_depth=6,
    verbose=2,
    random_state=42)

lmart_xgb_pipe = fbr >> pt.ltr.apply_learned_model(lmart_x, form="ltr")
lmart_xgb_pipe.fit(train_topics, train_qrels, validation_topics, validation_qrels)


# Experiment


systems = [TF_IDF, BM25, PL2, rf_pipe, lr_pipe, svr_pipe, lmart_xgb_pipe]
names  = ["TF-IDF", "BM25", "PL2", "Random Forest", "Logistic Regression", "Support Vector Regression", "LambdaMART"]

results = pt.Experiment(
    systems,
    test_topics,
    test_qrels,
    eval_metrics=["map", "ndcg", "P_20", "ndcg_cut_20"],
    baseline=0,
    names=names,
    correction='bonferroni',
    verbose=True)





