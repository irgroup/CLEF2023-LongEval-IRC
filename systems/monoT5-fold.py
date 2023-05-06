#  5 Fold cross validate
from src.exp_logger import logger
from src.load_index import setup_system
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import random
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
import yaml  # type: ignore
from pyterrier_t5 import MonoT5ReRanker

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

import pandas as pd

with open("settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)




class MonoT5Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f'Query: {sample[0]} Document: {sample[1]} Relevant:'
        return {
          'text': text,
          'labels': sample[2],
        }




def load_data(train, test, topics, qrels):
    # Load data
    relevant = pd.read_json("data/passages/t5/WT-relevant-passages.jsonl", lines=True)
    not_relevant = pd.read_json("data/passages/t5/WT-not-relevant-passages.jsonl", lines=True)

    ## topics
    train_topics = topics.iloc[train]
    test_topics = topics.iloc[test]
    
    ## qrels
    test_qrels = qrels[qrels["qid"].isin(test_topics["qid"])]

    ## passages
    train_relevant = relevant[relevant["qid"].isin(train_topics["qid"])]
    train_not_relevant = not_relevant[not_relevant["qid"].isin(train_topics["qid"])]

    ## samples
    train_samples = []
    train_relevant = train_relevant.merge(train_topics, on="qid", how="left")
    train_relevant["sample"] = train_relevant.apply(lambda x: [x["query"], x["passage"], "true"], axis=1)
    train_samples.extend(train_relevant["sample"].to_list())

    train_not_relevant = train_not_relevant.merge(train_topics, on="qid", how="left")
    train_not_relevant["sample"] = train_not_relevant.apply(lambda x: [x["query"], x["passage"], "false"], axis=1)
    train_samples.extend(train_not_relevant["sample"].to_list())
    
    ## shuffle
    random.Random(42).shuffle(train_samples)
    return train_samples, test_topics, test_qrels



def fit_model(samples):
    # cuda
    device = torch.device('cuda')
    torch.manual_seed(123)
    
    # settings
    base_model = "castorini/monoT5-base-msmarco"
    output_model_path = "data/models/monoT5-fold/checkpoints/"
    save_every_n_steps = 1000
    logging_steps = 100
    per_device_train_batch_size = 6
    gradient_accumulation_steps = 16
    learning_rate = 3e-4  # original
    epochs = 10

    # model
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    train_args = Seq2SeqTrainingArguments(
        output_dir=output_model_path,
        do_train=True,
        save_strategy="steps",
        save_steps =save_every_n_steps, 
        logging_steps=logging_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=5e-5,
        num_train_epochs=1,
        warmup_steps=0,
        # warmup_steps=1000,
        adafactor=True,
        seed=1,
        disable_tqdm=False,
        load_best_model_at_end=False,
        predict_with_generate=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    def smart_batching_collate_text_only(batch):
        texts = [example['text'] for example in batch]
        tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']

        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)

        return tokenized
    

    dataset_train = MonoT5Dataset(samples)

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=smart_batching_collate_text_only,
    )

    trainer.train()

    trainer.save_model(output_model_path.replace("checkpoints", "monoT5"))
    trainer.save_state()



def get_system(index):
    model_path = "data/models/monoT5-fold/monoT5/"

    bm25 = pt.BatchRetrieve(
        index, wmodel="BM25", verbose=True, metadata=["docno", "text"]
    ).parallel(6)

    monoT5 = MonoT5ReRanker(verbose=True, batch_size=8, model=model_path)
    
    mono_pipeline = bm25 >> pt.text.get_text(index, "text") >>  monoT5 
    return mono_pipeline


def main():
    index, topics, qrels = setup_system("WT")

    # sample topics just in case
    topics = topics.sample(frac=1, random_state=42).reset_index(drop=True)


    kf = KFold(n_splits=5)
    c = 0
    for train, test in kf.split(topics):
        # Load data
        train_samples, test_topics, test_qrels = load_data(train, test, topics, qrels)
        print(len(train_samples), len(test_topics), len(test_qrels))

        # Fit model
        fit_model(train_samples)

        # Create run
        system = get_system(index)

        c+=1
        run_tag = "monoT5-f"+c

        pt.io.write_results(system(topics), config["results_path"]+"fold/" + run_tag)
        
        print("Done with run ", run_tag)


if __name__ == "__main__":
    main()



