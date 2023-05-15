#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E5 dense retrieval system through faiss.

Example:
    Create runs on the train topics of the given index::

        $ python -m systems.e5 --index WT_e5_small --train
    
    Create runs on the test topics of the given index::

        $ python -m systems.e5 --index WT_e5_small
"""
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import yaml
import os
import json
import pandas as pd

from src.exp_logger import logger
from argparse import ArgumentParser

import pyterrier as pt  # type: ignore
from src.metadata import get_metadata, write_metadata_yaml
from src.load_index import setup_system, tag
import torch 
import faiss
from tqdm import tqdm


with open("settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

# TODO fix checkpoint loading dynamically
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base')
model = AutoModel.from_pretrained('intfloat/e5-base')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_ = model.to(device)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.no_grad()
def calc_embeddings(texts, mode='passage'):
  input_texts = [f"{mode}: {text}" for text in texts]
  batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
  for key, _ in batch_dict.items():
    batch_dict[key] = batch_dict[key].cuda(non_blocking=True)
  
  outputs = model(**batch_dict)
  embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
  return embeddings.detach().cpu()


def load_index(index_dir):
    """load faiss index"""
    index = faiss.read_index(index_dir+"/index")
    return index


def write_trec(run_tag, topics, I, D, ids):
    """write results as trec"""
    with open("results/trec/"+run_tag, "w") as f:
        for qid, query, results in zip(topics["qid"].to_list(), I, D):
            for rank, (doc_id, distance) in enumerate(zip(query, results)):
                f.write("{} Q0 {} {} {} IRC-e5\n".format(qid, ids[str(doc_id)], rank, 100-distance, run_tag))


def main(args):
    run_tag = tag(args.index[3:], args.index[:2])
    checkpoint = f"intfloat/{args.index[3:]}"

    slice = "train" if args.train else "test"
    topics = pt.io.read_topics(config[args.index][slice]["topics"])
    query_embedding = calc_embeddings(topics["query"], mode='query')
    with open(f"data/index/{args.index}/{args.index}_ids.json", "r") as f:
        ids = json.load(f)

    index = load_index(os.path.join(config["index_dir"], args.index))
    D, I = index.search(query_embedding.numpy(), k = 1000)

    write_trec(run_tag, topics, I, D, ids)
    
    write_metadata_yaml(
        config["metadata_path"] + run_tag + ".yml",
        {
            "tag": run_tag,
            "method": {
                "indexing": {
                    "e5": {
                        "method": "src.create_index_e5",
                        "checkpoint": checkpoint,
                        "index": "faiss.IndexFlatL2"
                    }
                },
                "retrieval": {
                    "1": {
                        "name": "knn",
                        "method": "faiss.IndexFlatL2.search",
                        "similarioty measure": "euclidean distance"
                    }
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
