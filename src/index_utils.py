import pyterrier as pt
import os
from typing import Union, Any, Dict
from config import CONFIG, INDEX_DIR

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
