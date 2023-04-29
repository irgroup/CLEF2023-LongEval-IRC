#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BM25 baseline systems including pseudo relevance feedback and rank fusion.

Example:
    Run the system with the following command::

        $ python -m systems.BM25 --index WT
"""
from argparse import ArgumentParser
from src.exp_logger import logger

import pyterrier as pt  # type: ignore
from ranx import Run, fuse

from src.load_index import setup_system

logger.setLevel("INFO")


def main(args):
    index, topics, _ = setup_system(args.index)

    BM25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True).parallel(6)
    pt.io.write_results(BM25(topics), "results/trec/IRCologne-BM25." + args.index)

    TF_IDF = pt.BatchRetrieve(index, wmodel="TF_IDF", verbose=True).parallel(6)
    pt.io.write_results(
        TF_IDF(topics),
        "results/trec/IRCologne-TF_IDF." + args.index,
        run_name="IRC-TF_IDF." + args.index,
    )

    XSqrA_M = pt.BatchRetrieve(index, wmodel="XSqrA_M", verbose=True).parallel(6)
    pt.io.write_results(
        XSqrA_M(topics),
        "results/trec/IRCologne-XSqrA_M." + args.index,
        run_name="IRC-XSqrA_M." + args.index,
    )

    PL2 = pt.BatchRetrieve(index, wmodel="PL2", verbose=True).parallel(6)
    pt.io.write_results(
        PL2(topics),
        "results/trec/IRCologne-PL2." + args.index,
        run_name="IRC-PL2." + args.index,
    )

    DPH = pt.BatchRetrieve(index, wmodel="DPH", verbose=True).parallel(6)
    pt.io.write_results(
        DPH(topics),
        "results/trec/IRCologne-DPH." + args.index,
        run_name="IRC-DPH." + args.index,
    )

    # Pseudo relevance feedback
    rm3_pipe = BM25 >> pt.rewrite.RM3(index) >> BM25
    pt.io.write_results(
        rm3_pipe(topics),
        "results/trec/IRCologne-BM25_RM3." + args.index,
        run_name="IRC-BM25_RM3." + args.index,
    )

    bo1_pipe = BM25 >> pt.rewrite.Bo1QueryExpansion(index) >> BM25
    pt.io.write_results(
        bo1_pipe(topics),
        "results/trec/IRCologne-BM25_Bo1." + args.index,
        run_name="IRC-BM25_Bo1." + args.index,
    )

    axio_pipe = BM25 >> pt.rewrite.AxiomaticQE(index) >> BM25
    pt.io.write_results(
        axio_pipe(topics),
        "results/trec/IRCologne-BM25_axio." + args.index,
        run_name="IRC-BM25_axio." + args.index,
    )

    # fuse: BM25, XSqrA_M, PL2
    baseline_ranx = Run.from_file(
        "results/trec/IRCologne-BM25." + args.index, kind="trec"
    )
    baseline_ranx.name = "BM25"

    runs = []
    runs.append(Run.from_file("results/trec/IRCologne-BM25." + args.index, kind="trec"))
    runs.append(
        Run.from_file("results/trec/IRCologne-XSqrA_M." + args.index, kind="trec")
    )
    runs.append(Run.from_file("results/trec/IRCologne-PL2." + args.index, kind="trec"))

    fuse_method = "rrf"

    run_rrf = fuse(runs=runs, method=fuse_method)
    run_rrf.name = "RRF_(BM25-XSqrA_M-PL2)"
    run_rrf.save("results/trec/IRCologne-RRF(BXP)." + args.index, kind="trec")

    # fuse: BM25-RM3, XSqrA_M, PL2

    baseline_ranx = Run.from_file(
        "results/trec/IRCologne-BM25." + args.index, kind="trec"
    )
    baseline_ranx.name = "BM25"

    runs = []
    runs.append(
        Run.from_file("results/trec/IRCologne-BM25_RM3." + args.index, kind="trec")
    )
    runs.append(
        Run.from_file("results/trec/IRCologne-XSqrA_M." + args.index, kind="trec")
    )
    runs.append(Run.from_file("results/trec/IRCologne-PL2." + args.index, kind="trec"))

    fuse_method = "rrf"

    run_rrf = fuse(runs=runs, method=fuse_method)
    run_rrf.name = "RRF_(BM25RM3-XSqrA_M-PL2)"
    run_rrf.save("results/trec/IRCologne-RRF(BRXP)." + args.index, kind="trec")

    # fuse: BM25-Bo1, XSqrA_M, PL2

    baseline_ranx = Run.from_file(
        "results/trec/IRCologne-BM25." + args.index, kind="trec"
    )
    baseline_ranx.name = "BM25"

    runs = []
    runs.append(
        Run.from_file("results/trec/IRCologne-BM25_Bo1." + args.index, kind="trec")
    )
    runs.append(
        Run.from_file("results/trec/IRCologne-XSqrA_M." + args.index, kind="trec")
    )
    runs.append(Run.from_file("results/trec/IRCologne-PL2." + args.index, kind="trec"))

    fuse_method = "rrf"

    run_rrf = fuse(runs=runs, method=fuse_method)
    run_rrf.name = "RRF_(BM25Bo1-XSqrA_M-PL2)"
    run_rrf.save("results/trec/IRCologne-RRF(BBXP)." + args.index, kind="trec")


if __name__ == "__main__":
    parser = ArgumentParser(description="Create an pyterrier index from a config.")
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Name of the dataset in the config file (WT, ST or LT)",
    )

    main(parser.parse_args())
