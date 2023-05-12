
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

from src.load_index import setup_system, tag
import torch
import faiss
from tqdm import tqdm


with open("settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)



tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base')
model = AutoModel.from_pretrained('intfloat/e5-base')
#prepare model for gpu use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_ = model.to(device)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.no_grad()
def calc_embeddings(texts, mode='passage'):
  input_texts = [f"{mode}: {text}" for text in texts]
  batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
  for key, val in batch_dict.items():
    batch_dict[key] = batch_dict[key].cuda(non_blocking=True)

  outputs = model(**batch_dict)
  embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
  return embeddings.detach().cpu()#.numpy()



def gen_docs(doc_path, batch_size):
    """Generate batches of documents from the WT collection. Creats a global dict of ids to doc ids."""
    global c
    c = 0
    global ids
    ids = {}
    batch = []
    for filename in os.listdir(doc_path):
        with open(doc_path+"/"+filename, "r") as f:
            for line in f:
                l = json.loads(line)
                for doc in l:
                    if len(batch) == batch_size:
                        full_batch = batch
                        batch  = []
                        batch.append(doc["contents"])
                        yield full_batch
                    else:
                        batch.append(doc["contents"])
                    ids[c]= doc["id"]
                    c+=1


def encode(doc_path, index_path, index, batch_size, num_docs, save_every, stop_at=10):
    """create embeddings for docs in batches and save in batches"""
    def save_embs(embs, c, batch_size):
        embs = torch.cat(embs)
        torch.save(embs, f"{index_path}/{index}/e5_embeddings_{c}.pt")
        logger.info(f"Saved embeddings for {c*batch_size} documents")

    def save_ids(ids):
        with open(os.path.join(index_path, index, index+"_ids.json"), "w") as f:
            json.dump(ids, f)
    c = 0
    embs = []
    for batch in tqdm(gen_docs(doc_path=doc_path, batch_size=batch_size), total=(int(num_docs/batch_size))):
        embeddings = calc_embeddings(batch)
        embs.append(embeddings)

        if len(embs) >= save_every:
            save_embs(embs, c, batch_size)
            c+=1
            embs = []
        if stop_at:
            if c == stop_at:
                break
    if embs:
        save_embs(embs, c, save_every)
    save_ids(ids)
    logger.info(f"Done with encoding")



# load index
def create_index(index_dir, size=768):
    """create index from embedding parts"""
    files = os.listdir(index_dir)
    files.sort()  # TODO sorting fails, to leading 0

    index = faiss.IndexFlatL2(size)   # build the index

    for file in files:
        if file.endswith(".pt"):
            index.add(torch.load(index_dir+"/"+file).numpy())
    faiss.write_index(index, index_dir+"/index")


def main(args):
    # Topics
    logger.setLevel("INFO")

    doc_path = config[args.index]["docs"].replace("Trec", "Json")

    if not os.path.exists(config["index_dir"]+args.index):
        os.makedirs(config["index_dir"]+args.index)

    encode(
        doc_path=doc_path, 
        index_path=config["index_dir"],
        index = args.index,
        batch_size=args.batch_size, 
        num_docs=1570734, 
        save_every=args.save
        )

    logger.info("Done with encodng, start indexing...")
    create_index(os.path.join(config["index_dir"]+args.index))
    logger.info("Done with indexing")


if __name__ == "__main__":
    parser = ArgumentParser(description="Create an pyterrier index from a config.")

    # input arguments
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Name of the dataset in the config file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--save",
        type=int,
        required=True,
        help="Save every x batches",
    )
    args = parser.parse_args()
    main(args)
