import sys

sys.path.append("..")
sys.path.append("/code")

from tira.third_party_integrations import (ensure_pyterrier_is_loaded,
                                           persist_and_normalize_run)

ensure_pyterrier_is_loaded()

import os
from argparse import ArgumentParser

import pandas as pd
from trectools import TrecEval, TrecQrel, TrecRun, procedures


def load_runs_to_route(dir):
    ret = []
    for run_name in os.listdir(dir):
        run = pd.read_csv(
            os.path.join(dir, run_name),
            sep=" ",
            names=["qid", "q0", "docno", "rank", "score", "system"],
        )
        ret.append(run)
        print(run["system"][0])
    return pd.concat(ret)


def load_qrels(qrels_file):
    return pd.read_csv(qrels_file, names=["query", "Q0", "docid", "rel"], sep=" ")


def qrels_for_query(qid):
    ground_truth = qrels[qrels["query"] == qid]
    ret = TrecQrel()
    ret.qrels_data = ground_truth
    return ret


def get_run_by_system(qid, system, all_runs):
    run = all_runs[(all_runs["qid"] == qid) & (all_runs["system"] == system)].copy()
    run = run.rename(columns={"query": "query_string"})
    run = run.rename(columns={"qid": "query", "docno": "docid"})
    ret = TrecRun()
    ret.run_data = run
    return ret


def calculate_score_of_system_on_query(qid, all_runs, system, measure="NDCG_5"):
    try:
        qrels = qrels_for_query(qid)
        run = get_run_by_system(qid, system, all_runs)

        ret = TrecEval(run, qrels).get_ndcg(depth=10, removeUnjudged=True)
    except:
        print(qid)
        return -1
    return ret


def select_best_system_per_query(qid, all_runs, measure="NDCG_5"):
    systems = all_runs["system"].unique()
    best_system = None
    best_score = None

    for system in systems:
        score_of_system = calculate_score_of_system_on_query(qid, all_runs, system)
        if best_system is None or best_score < score_of_system:
            best_system = system
            best_score = score_of_system
    return best_system


def build_run(all_runs):
    ret = []
    for qid in all_runs["qid"].unique():
        best_system = select_best_system_per_query(qid, all_runs)
        ret.append(
            all_runs[
                (all_runs["qid"] == qid) & (all_runs["system"] == best_system)
            ].copy()
        )

    return pd.concat(ret)


def main():
    parser = ArgumentParser(description="Route queries to systems")

    parser.add_argument(
        "--input_runs", default="images/test_runs", help="Path to input runs directory"
    )
    parser.add_argument(
        "--input_qrels",
        default="data/longeval-relevance-judgements/a-short-july.txt",
        help="Path to input qrels file",
    )
    parser.add_argument(
        "--output",
        default="./",
    )

    args = parser.parse_args()

    all_runs = load_runs_to_route(args.input_runs)
    global qrels
    qrels = load_qrels(args.input_qrels)

    routed_run = build_run(all_runs)

    persist_and_normalize_run(routed_run, "IRC", args.output)


if __name__ == "__main__":
    main()
