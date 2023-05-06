#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BM25 baseline systems including pseudo relevance feedback and rank fusion.

Example:
    Create runs on the train topics of the given index::

        $ python -m systems.BM25 --index WT --train
    
    Create runs on the test topics of the given index::

        $ python -m systems.BM25 --index WT
"""
from argparse import ArgumentParser
from src.exp_logger import logger

import pyterrier as pt  # type: ignore
from ranx import Run, fuse

from src.load_index import setup_system, tag
from src.metadata import get_metadata, write_metadata_yaml

logger.setLevel("INFO")


results_path = "results/trec/"
metadata_path = "results/metadata/"


def main(args):
    index, topics, _ = setup_system(args.index, train=args.train)

    # BM25
    run_tag = tag("BM25", args.index)
    BM25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True).parallel(6)
    pt.io.write_results(BM25(topics), results_path + run_tag)
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "bm25",
                        "method": "org.terrier.matching.models.BM25",
                        "k_1": "1.2",
                        "k_3": "8",
                        "b": "0.75",
                    }
                },
            },
        },
    )

    # TF_IDF
    run_tag = tag("TF_IDF", args.index)
    TF_IDF = pt.BatchRetrieve(index, wmodel="TF_IDF", verbose=True).parallel(6)
    pt.io.write_results(TF_IDF(topics), results_path + run_tag, run_name=run_tag)
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "TF_IDF",
                        "method": "org.terrier.matching.models.TF_IDF",
                    }
                },
            },
        },
    )

    # XSqrA_M
    run_tag = tag("XSqrA_M", args.index)
    XSqrA_M = pt.BatchRetrieve(index, wmodel="XSqrA_M", verbose=True).parallel(6)
    pt.io.write_results(XSqrA_M(topics), results_path + run_tag, run_name=run_tag)
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "XSqrA_M",
                        "method": "org.terrier.matching.models.XSqr_A_M",
                    }
                },
            },
        },
    )

    # PL2
    run_tag = tag("PL2", args.index)
    PL2 = pt.BatchRetrieve(index, wmodel="PL2", verbose=True).parallel(6)
    pt.io.write_results(PL2(topics), results_path + run_tag, run_name=run_tag)
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "PL2",
                        "method": "org.terrier.matching.models.PL2",
                    }
                },
            },
        },
    )

    # DPH
    run_tag = tag("DPH", args.index)
    DPH = pt.BatchRetrieve(index, wmodel="DPH", verbose=True).parallel(6)
    pt.io.write_results(DPH(topics), results_path + run_tag, run_name=run_tag)
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "DPH",
                        "method": "org.terrier.matching.models.DPH",
                    }
                },
            },
        },
    )

    # Pseudo relevance feedback
    # BM25 + RM3
    run_tag = tag("BM25+RM3", args.index)
    rm3_pipe = BM25 >> pt.rewrite.RM3(index) >> BM25
    pt.io.write_results(rm3_pipe(topics), results_path + run_tag, run_name=run_tag)
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "bm25",
                        "method": "org.terrier.matching.models.BM25",
                        "k_1": "1.2",
                        "k_3": "8",
                        "b": "0.75",
                    },
                    "2": {
                        "name": "RM3 query expansion",
                        "method": "pyterrier.rewrite.RM3",
                        "fb_terms": "10",
                        "fb_docs": "3",
                        "fb_lambda": "0.6",
                        "reranks": "bm25",
                    },
                },
            },
        },
    )

    # BM25 + Bo1
    run_tag = tag("BM25+Bo1", args.index)
    bo1_pipe = BM25 >> pt.rewrite.Bo1QueryExpansion(index) >> BM25
    pt.io.write_results(bo1_pipe(topics), results_path + run_tag, run_name=run_tag)
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "bm25",
                        "method": "org.terrier.matching.models.BM25",
                        "k_1": "1.2",
                        "k_3": "8",
                        "b": "0.75",
                    },
                    "2": {
                        "name": "Bo1 query expansion",
                        "method": "pyterrier.rewrite.Bo1QueryExpansion",
                        "fb_terms": "10",
                        "fb_docs": "3",
                        "reranks": "bm25",
                    },
                },
            },
        },
    )

    # BM25 + Axiomatic QE
    run_tag = tag("BM25+axio", args.index)
    axio_pipe = BM25 >> pt.rewrite.AxiomaticQE(index) >> BM25
    pt.io.write_results(axio_pipe(topics), results_path + run_tag, run_name=run_tag)
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "bm25",
                        "method": "org.terrier.matching.models.BM25",
                        "k_1": "1.2",
                        "k_3": "8",
                        "b": "0.75",
                    },
                    "2": {
                        "name": "axiomatic query expansion",
                        "method": "pyterrier.rewrite.AxiomaticQE",
                        "fb_terms": "10",
                        "fb_docs": "3",
                        "reranks": "bm25",
                    },
                },
            },
        },
    )

    # fuse: BM25, XSqrA_M, PL2
    run_tag = tag("RRF(BM25-XSqrA_M-PL2)", args.index)

    baseline_ranx = Run.from_file("results/trec/IRC-BM25." + args.index, kind="trec")
    baseline_ranx.name = "BM25"

    runs = []
    runs.append(Run.from_file("results/trec/IRC-BM25." + args.index, kind="trec"))
    runs.append(Run.from_file("results/trec/IRC-XSqrA_M." + args.index, kind="trec"))
    runs.append(Run.from_file("results/trec/IRC-PL2." + args.index, kind="trec"))

    fuse_method = "rrf"

    run_rrf = fuse(runs=runs, method=fuse_method)
    run_rrf.name = run_tag
    run_rrf.save(results_path + run_tag, kind="trec")
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "bm25",
                        "method": "org.terrier.matching.models.BM25",
                        "k_1": "1.2",
                        "k_3": "8",
                        "b": "0.75",
                    },
                    "2": {
                        "name": "XSqrA_M",
                        "method": "org.terrier.matching.models.XSqr_A_M",
                    },
                    "3": {
                        "name": "PL2",
                        "method": "org.terrier.matching.models.PL2",
                    },
                    "4": {
                        "name": "Reciprocal Rank Fusion (RRF)",
                        "method": "ranx.fusion.rrf",
                        "min_k": "10",
                        "max_k": "100",
                        "step": "10",
                        "fuses": [
                            "bm25",
                            "XSqrA_M",
                            "PL2",
                        ],
                    },
                },
            },
        },
    )

    # fuse: BM25-RM3, XSqrA_M, PL2
    run_tag = tag("RRF(BM25RM3-XSqrA_M-PL2)", args.index)
    baseline_ranx = Run.from_file("results/trec/IRC-BM25." + args.index, kind="trec")
    baseline_ranx.name = "BM25"

    runs = []
    runs.append(Run.from_file("results/trec/IRC-BM25+RM3_opt." + args.index, kind="trec"))
    runs.append(Run.from_file("results/trec/IRC-XSqrA_M." + args.index, kind="trec"))
    runs.append(Run.from_file("results/trec/IRC-PL2." + args.index, kind="trec"))

    fuse_method = "rrf"

    run_rrf = fuse(runs=runs, method=fuse_method)
    run_rrf.name = run_tag
    run_rrf.save(results_path + run_tag, kind="trec")
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "bm25",
                        "method": "org.terrier.matching.models.BM25",
                        "k_1": "1.2",
                        "k_3": "8",
                        "b": "0.75",
                    },
                    "2": {
                        "name": "RM3 query expansion",
                        "method": "pyterrier.rewrite.RM3",
                        "fb_terms": "10",
                        "fb_docs": "3",
                        "fb_lambda": "0.6",
                        "reranks": "bm25",
                    },
                    "3": {
                        "name": "XSqrA_M",
                        "method": "org.terrier.matching.models.XSqr_A_M",
                    },
                    "4": {
                        "name": "PL2",
                        "method": "org.terrier.matching.models.PL2",
                    },
                    "5": {
                        "name": "Reciprocal Rank Fusion (RRF)",
                        "method": "ranx.fusion.rrf",
                        "min_k": "10",
                        "max_k": "100",
                        "step": "10",
                        "fuses": [
                            "RM3 query expansion",
                            "XSqrA_M",
                            "PL2",
                        ],
                    },
                },
            },
        },
    )

    # fuse: BM25-Bo1, XSqrA_M, PL2
    run_tag = tag("RRF(BM25RM3-XSqrA_M-PL2)", args.index)
    baseline_ranx = Run.from_file("results/trec/IRC-BM25." + args.index, kind="trec")
    baseline_ranx.name = "BM25"

    runs = []
    runs.append(Run.from_file("results/trec/IRC-BM25+Bo1." + args.index, kind="trec"))
    runs.append(Run.from_file("results/trec/IRC-XSqrA_M." + args.index, kind="trec"))
    runs.append(Run.from_file("results/trec/IRC-PL2." + args.index, kind="trec"))

    fuse_method = "rrf"

    run_rrf = fuse(runs=runs, method=fuse_method)
    run_rrf.name = run_tag
    run_rrf.save("results/trec/IRC-RRF(BBXP)." + args.index, kind="trec")
    write_metadata_yaml(
        metadata_path + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "retrieval": {
                    "1": {
                        "name": "bm25",
                        "method": "org.terrier.matching.models.BM25",
                        "k_1": "1.2",
                        "k_3": "8",
                        "b": "0.75",
                    },
                    "2": {
                        "name": "Bo1 query expansion",
                        "method": "pyterrier.rewrite.Bo1QueryExpansion",
                        "fb_terms": "10",
                        "fb_docs": "3",
                        "reranks": "bm25",
                    },
                    "3": {
                        "name": "XSqrA_M",
                        "method": "org.terrier.matching.models.XSqr_A_M",
                    },
                    "4": {
                        "name": "PL2",
                        "method": "org.terrier.matching.models.PL2",
                    },
                    "5": {
                        "name": "Reciprocal Rank Fusion (RRF)",
                        "method": "ranx.fusion.rrf",
                        "min_k": "10",
                        "max_k": "100",
                        "step": "10",
                        "fuses": [
                            "RM3 query expansion",
                            "XSqrA_M",
                            "PL2",
                        ],
                    },
                },
            },
        },
    )


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
        help="Use the train topics to create the.",
    )

    args = parser.parse_args()
    main(args)
