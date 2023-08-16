import os
import sys
sys.path.append("..")
sys.path.append("/code")

from tira.third_party_integrations import persist_and_normalize_run, ensure_pyterrier_is_loaded

ensure_pyterrier_is_loaded()

from create_index_e5 import calc_embeddings, load_model
import pandas as pd
from torch.nn import CosineSimilarity
from torch import dot
from argparse import ArgumentParser 


def load_queries_to_rerank(dir):
    df = pd.read_json(f'{dir}/rerank.jsonl.gz', lines = True)
    return df[["qid", "query"]].drop_duplicates()


def load_documents_to_rerank(dir, qid):
    df = pd.read_json(f'{dir}/rerank.jsonl.gz', lines = True)
    return df[df["qid"].astype(str)==qid][["docno", "text"]]


def encode_queries(queries):
    return list(zip(queries["qid"], calc_embeddings(list(queries["query"]), "query")))


def encode_documents(documents):
    return list(zip(documents["docno"], calc_embeddings(list(documents["text"]))))


def rerank_for_query(qid, query_vector, dir, sim_func):
    documents = load_documents_to_rerank(dir, str(qid))
    document_vectors = encode_documents(documents)
    ranking = []
    for docno, doc_vector in document_vectors:
        ranking.append({"qid": qid, "docno": docno, "score": sim_func(doc_vector, query_vector)})
    return pd.DataFrame(ranking)


def cosine_similarity(x, y):
    return CosineSimilarity(dim=0)(x, y).item()

def dot_product(x, y):
    return dot(x, y).item()

def main():

    parser = ArgumentParser(description="rerank with E5")
    parser.add_argument(
        "--similarity",
        choices=["cos", "dot"],
        default="cos",
        help="cos for cosinus similarity or dot for dot product.",
    )
    parser.add_argument(
        "--input",
        default="../images/rerank_dataset",
        help="Path to dataset"
    )
    parser.add_argument(
        "--output",
        default="/tmp",
    )


    args = parser.parse_args()


    load_model('/model/E5-base', 'intfloat/e5-base')

    queries = load_queries_to_rerank(args.input)
    query_vectors = encode_queries(queries)

    similarity = cosine_similarity if args.similarity == "cos" else dot_product

    run = []
    for qid, query_vector in query_vectors:
        run.append(rerank_for_query(qid, query_vector, args.input, similarity))

    persist_and_normalize_run(pd.concat(run), "IRC", args.output)

if __name__ == "__main__":
    main()