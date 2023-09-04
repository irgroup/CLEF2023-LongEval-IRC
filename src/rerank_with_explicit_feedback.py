import pyterrier as pt  # type: ignore
import os
import yaml
from tira.third_party_integrations import (
    persist_and_normalize_run,
    ensure_pyterrier_is_loaded,
)
from pt.rewrite import *
from argparse import ArgumentParser

import pandas as pd
from trectools import TrecRun, TrecQrel, TrecEval

from trectools import TrecQrel, procedures

ensure_pyterrier_is_loaded()


def load_query_translation(target_corpus):
    ret = pd.read_csv("data/core_queries.tsv", sep="\t")
    return {i[f"qid_{target_corpus}"]: i["qid_WT"] for _, i in ret.iterrows()}


def load_doc_translation():
    ret = pd.read_csv("data/core_docs.tsv", sep="\t")
    return {
        i["docno_WT"]: {"WT": i["docno_WT"], "LT": i["docno_LT"], "ST": i["docno_ST"]}
        for _, i in ret.iterrows()
    }


def sorted_training_data_for_query(qid):
    ret_train = pd.read_csv(
        "data/publish/French/Qrels/train.txt",
        names=["qid", "Q0", "docno", "relevance"],
        sep=" ",
    )
    ret_test = pd.read_csv(
        "data/longeval-relevance-judgements/heldout-test.txt",
        names=["qid", "Q0", "docno", "relevance"],
        sep=" ",
    )

    ret = pd.concat([ret_train, ret_test])
    return ret[ret["qid"].astype(str) == qid].sort_values("relevance", ascending=False)


def ground_truth_ranking_from_training_data(qid):
    mapping = {"q06": "WT", "q07": "ST", "q09": "LT"}
    target_corpus = mapping[qid[:3]]
    query_mapping = load_query_translation(target_corpus)

    training_data = sorted_training_data_for_query(query_mapping[qid])

    training_data["translated"] = training_data["docno"].apply(
        lambda i: doc_translation.get(i, {}).get(target_corpus, None)
    )
    training_data = training_data.dropna()
    return pd.DataFrame(
        [
            {"qid": qid, "docno": i["translated"], "score": i["relevance"]}
            for _, i in training_data.iterrows()
        ]
    )


def ground_truth_ranking_from_training_data_transformer(df):
    qids = df["qid"].unique()
    qid_to_query = {i["qid"]: i["query"] for _, i in df.iterrows()}

    ret = []
    for qid in qids:
        ranking = ground_truth_ranking_from_training_data(qid)

        if len(ranking) > 0:
            ranking["query"] = ranking["qid"].apply(lambda i: qid_to_query[i])
        ret.append(ranking)
    return pd.concat(ret)


def build_pipeline(index, expansion_model, wmodel):

    explicit_relevance_feedback = pt.apply.generic(
        ground_truth_ranking_from_training_data_transformer
    )
    query_expansion = globals()[expansion_model](index)

    retrieval = pt.BatchRetrieve(index, wmodel=wmodel)

    ret = explicit_relevance_feedback >> query_expansion >> retrieval
    return ret


def load_queries(dir):
    return pt.IndexFactory.of(dir)


def load_documents(dir):
    return pt.io.read_topics(dir)


def main():
    parser = ArgumentParser(description="Rerank with explicit relevance feedback")

    parser.add_argument(
        "--input", default="../images/rerank_dataset", help="Path to dataset"
    )
    parser.add_argument(
        "--output",
        default="/tmp",
    )
    parser.add_argument(
        "--expansion_model",
        default="/tmp",
    )
    parser.add_argument(
        "--wmodel",
        default="/tmp",
    )
    args = parser.parse_args()

    global doc_translation
    doc_translation = load_doc_translation()

    index = load_queries(args.input)
    queries = load_documents(args.input)

    system = build_pipeline(index, expansion_model=args.expansion_model, wmodel=args.wmodel)

    run = system(queries)

    persist_and_normalize_run(run, "IRC", args.output)


if __name__ == "__main__":
    main()
