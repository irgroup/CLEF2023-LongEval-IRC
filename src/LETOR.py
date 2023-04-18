from typing import Union
import pandas as pd
import pyterrier as pt
import numpy as np

import logging

# create logger
logger = logging.getLogger('LETOR')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)



class LETOR:
    def __init__(self, index, query_path: str) -> None:
        # Doc index
        self.num_tokens = index.getCollectionStatistics().getNumberOfTokens()
        self.num_docs = index.getCollectionStatistics().getNumberOfDocuments()

        self.doc_di = index.getDirectIndex()
        self.doc_doi = index.getDocumentIndex()
        self.doc_lex = index.getLexicon()

        # Query index
        self.queries, self.qid_to_docno = self.prepare_query_index(query_path)
        self.query_index = self.index_queries(self.queries)
        
        self.query_di = self.query_index.getDirectIndex()
        self.query_doi = self.query_index.getDocumentIndex()
        self.query_lex = self.query_index.getLexicon()
        self.query_meta = self.query_index.getMetaIndex()



    ############## Doc index ##############
    def tf(self, token: str) -> int:
        tf = self.lexicon[token].getFrequency()
        logger.info(f"tf(`{token}`) = {tf}")
        return tf
    
    def df(self, token: str) -> int:
        df = self.lexicon[token].getDocumentFrequency()
        logger.info(f"df(`{token}`) = {df}")
        return df
    
    def idf(self, token: str) -> float:
        idf = np.log(self.num_docs / self.doc_lex[token].getDocumentFrequency())
        logger.info(f"idf(`{token}`) = {idf}")
        return idf




    # query index
    def prepare_query_index(self, query_path: str) -> Union[pd.DataFrame, dict[str, int]]:
        # prepare df
        queries = pt.io.read_topics(query_path)
        queries = queries.reset_index().rename(columns={"index": "docno"})
        queries["docno"] = queries["docno"].astype(str)  # docno must be a string?!

        # prepare patch dict
        qid_to_docno = dict(zip(queries["qid"], queries["docno"].astype(int)))

        return queries, qid_to_docno


    def index_queries(self, queries: pd.DataFrame) -> pt.index:    
        pd_indexer = pt.DFIndexer("./tmp", type=pt.IndexingType.MEMORY)
        indexref2 = pd_indexer.index(queries["query"], queries["docno"], queries["qid"])
        return pt.IndexFactory.of(indexref2)


    # get tokens
    def get_query_tokens(self, query_id: int) -> set[str]:
        """Use a separate index to get the full processing pipeline for the query."""
        query_tokens = set()
        index_id = self.qid_to_docno[query_id]
        posting = self.query_di.getPostings(self.query_doi.getDocumentEntry(index_id))
        for t in posting:
            stemm = self.query_lex.getLexiconEntry(t.getId()).getKey()
            query_tokens.add(stemm)
        return query_tokens


    # get doc tokens
    def get_doc_tokens(self, doc_id: int) -> set[str]:
        doc_tokens = set()
        posting = self.doc_di.getPostings(self.doc_doi.getDocumentEntry(doc_id))
        for t in posting:
            stemm = self.doc_lex.getLexiconEntry(t.getId()).getKey()
            doc_tokens.add(stemm)
        return doc_tokens
    

    ########### Query ###########
    def get_doc_tf(self, query_id, doc_id) -> list[int]:
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
        logger.info(f"tf_idf = {tf_idf}")
        return tf_idf if tf_idf else [0]


    ############## Feature API ##############
    def get_features_letor(self, query_id: int, doc_id: int) -> int:
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
            
            sum(tfs)/stream_length,
            min(tfs)/stream_length,
            max(tfs)/stream_length,
            np.mean(tfs)/stream_length,
            np.var(tfs)/stream_length,

            # Tf-idf
            sum(tf_idfs),
            min(tf_idfs),
            max(tf_idfs),
            np.mean(tf_idfs),
            np.var(tf_idfs),

            # bool
            self.boolean_model_96(query_id, doc_id),

            # vector_space_model_101
            # BM25_106
            # LMIRABS_111
            # LMIRACLM_116
            # LMIRADIR_121

            # Number of slash in URL
            # Length of URL
            # Inlink number
            # Outlink number
            # PageRank
            # SiteRank
            # QualityScore
            # QualityScore2
            # Query-url click count
            # url click count
            # url dwell time
            ]
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

        logger.info(f"covered_query_term_number = {covered_query_term_number}")
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
        covered_query_term_ratio = self.query_doi.getDocumentLength(index_id) / self.doc_doi.getDocumentLength(doc_id)

        logger.info(f"covered_query_term_ratio = {covered_query_term_ratio}")
        return covered_query_term_ratio 

    def stream_length_11(self, doc_id: int) -> int:
        """Length of the document.

        Args:
            doc_id (int): Id of the document.

        Returns:
            int: length of the document.
        """
        stream_length = self.doc_doi.getDocumentLength(doc_id)
        logger.info(f"stream_length = {stream_length}")
        return stream_length

    ########## Idf ##########
    def idf_inverse_document_frequency_16(self, idfs: list[float]) -> float:
        """Sum of the inverse document frequency of the query terms.

        Args:
            idfs (list[float]): list of idfs.

        Returns:
            float: sum of the inverse document frequency of the query terms.
        """
        sum_of_idf = sum(idfs)
        logger.info(f"summed_query_idf = {sum_of_idf}")
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


