import pyterrier as pt
import os
from typing import Union, Any, Dict
from config import CONFIG, INDEX_DIR
from tqdm import tqdm
import pandas as pd


os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
if not pt.started():
    pt.init()


def create_index(index_name: str, documents_path: str):
    indexer = pt.TRECCollectionIndexer(
        INDEX_DIR + index_name,
        meta={"docno": 26, "text": 100000},
        meta_tags={"text": "ELSE"},
        blocks=True,
        verbose=True,
    )

    documents = [os.path.join(documents_path, path) for path in os.listdir(documents_path)]
    index = indexer.index(documents)

    return index


def load_index(index_name: str) -> pt.IndexFactory:
    index = pt.IndexFactory.of(INDEX_DIR + index_name)
    print(
        "Loaded index with ", index.getCollectionStatistics().getNumberOfDocuments(), "documents."
    )
    return index


def setup_system(system_config: Union[Dict[str, Any], str]):
    if isinstance(system_config, str):
        system_config = CONFIG[system_config]
    elif isinstance(system_config, dict):
        system_config = system_config

    index = load_index(system_config["index_name"])
    topics = pt.io.read_topics(system_config["topics"])
    qrels = pt.io.read_qrels(system_config["qrels"])
    return index, topics, qrels

def create_passeges(index_name: str):
    index, topics, qrels = setup_system(index_name)

    pipe = ( pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"])
        >> pt.text.sliding(length=128, stride=64, prepend_attr=None, text_attr="text") %1000
        >> pt.text.scorer(body_attr="text", wmodel="BM25"))


    for topic in tqdm(topics.iterrows(), total=len(topics)):
        # get passages
        qid = topic[1]["qid"]
        passages = pipe.transform(topics[topics["qid"]==qid])
        # Prpare passages
        passages["docno_full"] = passages["docno"].str.split("%").str[0]


        # prepare datasets
        passages_graded = passages.merge(qrels, left_on="docno_full", right_on="docno").sort_values("score", ascending=False)
        
        passages_rel=passages_graded[passages_graded["label"]==1][["docno_x", "text"]]
        passages_not_rel=passages_graded[passages_graded["label"]==0][["docno_x", "text"]]
        min_len = min([len(passages_rel), len(passages_not_rel)])

        id_list = [qid for _ in range(min_len)]

        triplets = pd.DataFrame(
            data={
            "qid": id_list, 
            "pid+": passages_rel["docno_x"].to_list()[:min_len], 
            "pid-": passages_not_rel["docno_x"].to_list()[:min_len],
            }
            )
        triplets.to_csv(f"data/passages/triplets/{qid}.csv", index=False, sep="\t", header=False)
        passages[["docno","text"]].to_csv(f"data/passages/colletion/{qid}.csv", index=False, sep="\t", header=False)