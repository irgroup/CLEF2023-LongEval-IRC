#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LambdaMART systems trained on LETOR.

Example:
    Train a system on the train slice of the train topics and create a run on the test slice of the train topics ::

        $ python -m systems.LambdaMART-LGB_LETOR --index WT  --train
    
    Train a system on the train slice of the train topics and create a run on the test topics (submission run) ::

        $ python -m systems.LambdaMART-LGB_LETOR --index WT
"""
import json
import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from src.exp_logger import logger, get_new_logger  # type: ignore

import numpy as np
import pandas as pd  # type: ignore
import pyterrier as pt  # type: ignore
import yaml  # type: ignore
import lightgbm as lgb

from src.load_index import setup_system, tag, get_train_splits
from src.metadata import write_metadata_yaml

with open("settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

letor_logger = get_new_logger("letor")
caching_logger = get_new_logger("caching")


class LETOR:
    def __init__(
        self,
        index,
        query_path: str,
        url_path: str,
        caching: bool = True,
        cache_dir: str = "data/cache.jsonl",
    ) -> None:
        # Doc index
        self.num_tokens = index.getCollectionStatistics().getNumberOfTokens()
        self.num_docs = index.getCollectionStatistics().getNumberOfDocuments()

        self.doc_di = index.getDirectIndex()
        self.doc_doi = index.getDocumentIndex()
        self.doc_lex = index.getLexicon()

        # urls
        self.urls = self.load_urls(url_path)

        # Query index
        self.queries, self.qid_to_docno = self.prepare_query_index(query_path)
        self.query_index = self.index_queries(self.queries)

        self.query_di = self.query_index.getDirectIndex()
        self.query_doi = self.query_index.getDocumentIndex()
        self.query_lex = self.query_index.getLexicon()
        self.query_meta = self.query_index.getMetaIndex()

        # Cache
        self.caching = caching
        self.cache_dir = cache_dir
        self.cache = self.load_cache()

    def load_urls(self, url_path: str) -> pd.DataFrame:
        urls = {}
        with open(url_path, "r") as f:
            for line in f:
                docno, url = line.strip().split("\t")
                urls[docno] = url
        return urls

    def load_cache(self) -> Optional[Dict[str, List[Any]]]:
        if os.path.exists(self.cache_dir):
            cache = {}
            with open(self.cache_dir, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    cache_id = list(entry.keys())[0]
                    cache[cache_id] = entry[cache_id]
            caching_logger.info("Cache loaded")
            return cache
        else:
            caching_logger.warning("Cache not found")
            return None

    def write_to_cache(self, cache_line: Dict[str, List[Any]]) -> None:
        with open(self.cache_dir, "a+") as f:
            json.dump(cache_line, f)
            f.write("\n")

    ############## Doc index ##############
    def tf(self, token: str) -> int:
        tf = self.doc_lex[token].getFrequency()
        letor_logger.info(f"tf(`{token}`) = {tf}")
        return tf

    def df(self, token: str) -> int:
        df = self.doc_lex[token].getDocumentFrequency()
        letor_logger.info(f"df(`{token}`) = {df}")
        return df

    def idf(self, token: str) -> float:
        idf = np.log(self.num_docs / self.doc_lex[token].getDocumentFrequency())
        letor_logger.info(f"idf(`{token}`) = {idf}")
        return idf

    # query index
    def prepare_query_index(
        self, query_path: str
    ) -> Tuple[pd.DataFrame, Dict[int, int]]:
        # prepare df
        queries = pt.io.read_topics(query_path)
        queries = queries.reset_index().rename(columns={"index": "docno"})
        queries["docno"] = queries["docno"].astype(str)  # docno must be a string?!

        # prepare patch dict
        qid_to_docno = dict(zip(queries["qid"], queries["docno"].astype(int)))

        return queries, qid_to_docno

    def index_queries(self, queries: pd.DataFrame) -> pt.index:
        queries = queries.rename(columns={"query": "text"})
        iter_indexer = pt.IterDictIndexer(
            "./tmp", type=pt.IndexingType.MEMORY, meta={"docno": 20, "text": 4096}
        )
        indexref = iter_indexer.index(queries.to_dict("records"))

        return pt.IndexFactory.of(indexref)

    # get tokens
    def get_query_tokens(self, query_id: int) -> Set[str]:
        """Use a separate index to get the full processing pipeline for the query."""
        query_tokens = set()
        index_id = self.qid_to_docno[query_id]
        posting = self.query_di.getPostings(self.query_doi.getDocumentEntry(index_id))
        for t in posting:
            stemm = self.query_lex.getLexiconEntry(t.getId()).getKey()
            query_tokens.add(stemm)
        return query_tokens

    # get doc tokens
    def get_doc_tokens(self, doc_id: int) -> Set[str]:
        doc_tokens = set()
        posting = self.doc_di.getPostings(self.doc_doi.getDocumentEntry(doc_id))
        for t in posting:
            stemm = self.doc_lex.getLexiconEntry(t.getId()).getKey()
            doc_tokens.add(stemm)
        return doc_tokens

    ########### Query ###########
    def get_doc_tf(self, query_id, doc_id) -> List[int]:
        query_tokens = self.get_query_tokens(query_id)
        doc_tokens = self.get_doc_tokens(doc_id)
        relevant_token = query_tokens.intersection(doc_tokens)

        tf = []
        for posting in self.doc_di.getPostings(self.doc_doi.getDocumentEntry(doc_id)):
            termid = posting.getId()

            lee = self.doc_lex.getLexiconEntry(termid)

            if lee.getKey() in relevant_token:
                tf.append(posting.getFrequency())

        return tf if tf else [0]

    # get tf-idf for query and doc
    def tf_idf(self, tf, idf):
        tf_idf = [tf[i] * idf[i] for i in range(len(tf))]
        letor_logger.info(f"tf_idf = {tf_idf}")
        return tf_idf if tf_idf else [0]

    ############## Feature API ##############
    def get_features_letor(
        self, query_id: int, doc_id: int, docno: str
    ) -> List[Union[int, float]]:
        if self.caching and self.cache:
            features = self.cache.get(str(query_id) + "-" + str(doc_id))
            if features:
                caching_logger.info(
                    f"Cache hit for query '{query_id}' and doc '{doc_id}'"
                )
                return features

        # prepare stats
        tfs = self.get_doc_tf(query_id, doc_id)
        idfs = [self.idf(token) for token in self.get_query_tokens(query_id)]
        tf_idfs = self.tf_idf(tfs, idfs)

        stream_length = self.stream_length_11(doc_id)

        # prepare features
        features = [
            self.covered_query_term_number_1(query_id, doc_id),
            self.covered_query_term_ratio_6(query_id, doc_id),
            stream_length,
            self.idf_inverse_document_frequency_16(idfs),
            # Tf
            sum(tfs),
            min(tfs),
            max(tfs),
            np.mean(tfs),
            np.var(tfs),
            sum(tfs) / stream_length,
            min(tfs) / stream_length,
            max(tfs) / stream_length,
            np.mean(tfs) / stream_length,
            np.var(tfs) / stream_length,
            # Tf-idf
            sum(tf_idfs),
            min(tf_idfs),
            max(tf_idfs),
            np.mean(tf_idfs),
            np.var(tf_idfs),
            # bool
            self.boolean_model_96(query_id, doc_id),
            # url
            self.number_of_slash_in_url_126(docno),
            self.length_of_url_127(docno),
        ]
        if self.caching:
            caching_logger.info(f"Cache features for '{query_id}-{doc_id}'")
            self.write_to_cache({str(query_id) + "-" + str(doc_id): features})

        return features

    ####################
    ##### Features #####
    ####################

    ########## Term Coverage ##########
    def covered_query_term_number_1(self, query_id: int, doc_id: int) -> int:
        """Number of terms in the query that are also in the document.

        Args:
           query_id (int): Id of the query.
            doc_id (int): Id of the document.

        Returns:
            int: number covered query terms.
        """
        query_tokens = self.get_query_tokens(query_id)
        doc_tokens = self.get_doc_tokens(doc_id)

        covered_query_term_number = len(query_tokens.intersection(doc_tokens))

        letor_logger.info(f"covered_query_term_number = {covered_query_term_number}")
        return covered_query_term_number

    def covered_query_term_ratio_6(self, query_id: int, doc_id: int) -> float:
        """Ratio of terms in the query that are also in the document.

        Args:
           query_id (int): Id of the query.
            doc_id (int): Id of the document.

        Returns:
            float: ratio covered query terms.
        """
        index_id = self.qid_to_docno[query_id]
        covered_query_term_ratio = self.query_doi.getDocumentLength(
            index_id
        ) / self.doc_doi.getDocumentLength(doc_id)

        letor_logger.info(f"covered_query_term_ratio = {covered_query_term_ratio}")
        return covered_query_term_ratio

    def stream_length_11(self, doc_id: int) -> int:
        """Length of the document.

        Args:
            doc_id (int): Id of the document.

        Returns:
            int: length of the document.
        """
        stream_length = self.doc_doi.getDocumentLength(doc_id)
        letor_logger.info(f"stream_length = {stream_length}")
        return stream_length

    ########## Idf ##########
    def idf_inverse_document_frequency_16(self, idfs: List[float]) -> float:
        """Sum of the inverse document frequency of the query terms.

        Args:
            idfs (list[float]): list of idfs.

        Returns:
            float: sum of the inverse document frequency of the query terms.
        """
        sum_of_idf = sum(idfs)
        letor_logger.info(f"summed_query_idf = {sum_of_idf}")
        return sum_of_idf

    ########## boolean ##########
    def boolean_model_96(self, query_id: int, doc_id: int) -> int:
        """Boolean model.

        Args:
            query_id (int): Id of the query.
            doc_id (int): Id of the document.

        Returns:
            int: 1 if all query terms are in the document, 0 otherwise.
        """
        query_tokens = self.get_query_tokens(query_id)
        doc_tokens = self.get_doc_tokens(doc_id)
        covered_query_term_number = query_tokens.intersection(doc_tokens)

        if covered_query_term_number == query_tokens:
            return 1
        else:
            return 0

    def number_of_slash_in_url_126(self, docno) -> int:
        """Number of slashes in the URL.

        Args:
            query_id (int): Id of the query.
            doc_id (int): Id of the document.

        Returns:
            int: number of slashes in the URL.
        """
        url = self.urls[docno]
        number_of_slash_in_url = url.count("/")
        return number_of_slash_in_url

    def length_of_url_127(self, docno) -> int:
        """Length of the URL.

        Args:
            query_id (int): Id of the query.
            doc_id (int): Id of the document.

        Returns:
            int: length of the URL.
        """
        url = self.urls[docno]
        length_of_url = len(url)
        return length_of_url


def main(args):
    # Data for training (load train topics anyway)
    index, topics, qrels = setup_system(args.index, train=True)

    # Get qrels
    (
        train_topics,
        validation_topics,
        _,
        train_qrels,
        validation_qrels,
        _,
    ) = get_train_splits(topics, qrels)

    # Get features
    letor = LETOR(
        index,
        query_path=config[args.index]["train"]["topics"],
        url_path=config[args.index]["urls"],
    )

    def _features(row):
        docid = row["docid"]
        queryid = row["qid"]
        features = row["features"]  # get the features from WMODELs
        docno = row["docno"]

        # LETOR Features
        letor_features = letor.get_features_letor(queryid, docid, docno)

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

    lmart_l = lgb.LGBMRanker(
        task="train",
        min_data_in_leaf=1,
        min_sum_hessian_in_leaf=100,
        max_bin=255,
        num_leaves=7,
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5, 10],
        learning_rate=0.1,
        importance_type="gain",
        num_iterations=10,
    )

    lmart_l_pipe = fbr >> pt.ltr.apply_learned_model(lmart_l, form="ltr")
    lmart_l_pipe.fit(train_topics, train_qrels, validation_topics, validation_qrels)

    # Create Run
    if args.train:
        # reuse the train topics
        run_tag = tag("BM25+LambdaMART_LGB_LETOR_URL-train", "WT")
    else:
        # use the test topics
        _, topics, _ = setup_system(args.index, train=False)
        run_tag = tag("BM25+LambdaMART_LGB_LETOR_URL", "WT")

    pt.io.write_results(lmart_l_pipe(topics), config["results_path"] + run_tag)
    write_metadata_yaml(
        config["metadata_path"] + run_tag + ".yml",
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
                        "name": "LambdaMART Reranker",
                        "method": "lightgbm.sklearn.LGBRanker",
                        "min_data_in_leaf": 1,
                        "min_sum_hessian_in_leaf": 100,
                        "max_bin": 255,
                        "num_leaves": 7,
                        "objective": "lambdarank",
                        "metric": "ndcg",
                        "ndcg_eval_at": [1, 3, 5, 10],
                        "learning_rate": 0.1,
                        "importance_type": "gain",
                        "num_iterations": 10,
                        "reranks": "bm25",
                    },
                },
            },
        },
    )


if __name__ == "__main__":
    caching_logger.setLevel("INFO")

    parser = ArgumentParser(description="Run BM25+LambdaMART_LGB_LETOR")
    parser.add_argument(
        "--index",
        type=str,
        default="WT",
        help="Name of the index to be used.",
    )
    parser.add_argument(
        "--train",
        required=False,
        action="store_true",
        help="Use the train topics to create the.",
    )

    args = parser.parse_args()
    main(args)
