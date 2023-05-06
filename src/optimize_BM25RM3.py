#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate systems. 

If the system name is specified as in the compressed runs, the results are loaded from 
the saved run files and do not need to be computed again. 
"""
from src.exp_logger import logger  # type: ignore
from src.load_index import setup_system, get_train_splits
import pyterrier as pt  # type: ignore
import numpy as np
import yaml

with open("settings.yml", "r") as f:
    config = yaml.safe_load(f)


# from systems import ColBERT, ColBERT_LE


index, topics, qrels = setup_system("WT", train=True)
train_topics, validation_topics, test_topics, train_qrels, validation_qrels, test_qrels = get_train_splits(topics, qrels)


train_topics = train_topics[train_topics["qid"].isin(config["top_runs"])]


bm25_for_qe = pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.b" : 0.75})
rm3 = pt.rewrite.RM3(index, fb_terms=10, fb_docs=3)
pipe_qe = bm25_for_qe >> rm3 >> bm25_for_qe


param_map = {
        bm25_for_qe : { "bm25.b" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ]},
        rm3 : {
            "fb_terms" : list(range(1, 12, 3)), # makes a list of 1,3,6,7,12
            "fb_docs" : list(range(2, 30, 6))   # etc.
        }
}


pipe_qe = pt.GridSearch(pipe_qe, param_map, train_topics, train_qrels, metric="bpref", verbose=True)



with open("grid_results.txt", "w") as f:
    f.write(pipe_qe)
    f.write("####################################")
    f.write(pipe_qe.best_params_)


# Best bpref is 0.048931

# Best setting is ['BR(BM25) bm25.b=0.9', 'QueryExpansion(/home/jueri/dev/LongEval/data/index/index_WT/data.properties,26,10,<org.terrier.querying.RM3 at 0x7fdefa3009a0 jclass=org/terrier/querying/RM3 jself=<LocalRef obj=0x2a9f220 at 0x7fdef9e3d490>>) fb_terms=10', 'QueryExpansion(/home/jueri/dev/LongEval/data/index/index_WT/data.properties,26,10,<org.terrier.querying.RM3 at 0x7fdefa3009a0 jclass=org/terrier/querying/RM3 jself=<LocalRef obj=0x2a9f220 at 0x7fdef9e3d490>>) fb_docs=8']


# PyTerrier 0.9.2 has loaded Terrier 5.7 (built by craigm on 2022-11-10 18:30) and terrier-helper 0.0.7
# 
# No etc/terrier.properties, using terrier.default.properties for bootstrap configuration.
# Loaded index with  1570734 documents.
# GridScan:  45%|██████████████████████████████████████████████████████████████████████████▋                                                                                           | 99/220 [1:42:52<2:03:57, 61.47s/it]         GridScan: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 220/220 [3:37:17<00:00, 59.26s/it]
# Best map is 0.019001
# Best setting is ['BR(BM25) bm25.b=0.8', 'QueryExpansion(/home/juerikeller/dev/LongEval/data/index/index_WT/data.properties,26,10,<org.terrier.querying.RM3 at 0x7fe1078f3e50 jclass=org/terrier/querying/RM3 jself=<LocalRef obj=0x1a897a0 at 0x7fe1079d2570>>) fb_terms=10', 'QueryExpansion(/home/juerikeller/dev/LongEval/data/index/index_WT/data.properties,26,10,<org.terrier.querying.RM3 at 0x7fe1078f3e50 jclass=org/terrier/querying/RM3 jself=<LocalRef obj=0x1a897a0 at 0x7fe1079d2570>>) fb_docs=2']
# pt.Experiment: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [02:53<00:00, 173.34s/system]
# juerikeller@i


