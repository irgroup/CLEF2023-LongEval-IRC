#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate basic corpus statistics.

Statistics are calculated on document level: 
    - Stream lenghth, 
    - #token, 
    - #tokens without stopwords, 
    - #unique tokens without stopwords, 
    - #tokens stemmed
    - #unique tokens without stopwords  

TODO: run in parallel on files to save memory and increase performance

Example:
    Run the system with the following command::

        $ python -m src.data_stats --index WT
"""
import os
from argparse import ArgumentParser

import nltk  # type: ignore
import pandas as pd  # type: ignore

nltk.download("stopwords")
import yaml  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore

from src.exp_logger import logger

logger.setLevel("INFO")

with open("../settings.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def load_data(path: str) -> pd.DataFrame:
    logger.info("Start loading the dataset...")
    doc_table = pd.DataFrame()
    path = path.replace("Trec", "Json")
    for doc in os.listdir(path):
        doc
        df = pd.read_json(os.path.join(path, doc))
        doc_table = pd.concat([doc_table, df], ignore_index=True)
    logger.info("Dataset loaded")
    return doc_table


def clac_stats(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    # clean
    df[text_column] = df[text_column].str.replace("\n", " ")
    logger.info("Dataset cleaned")

    # stream
    df["len stream"] = df[text_column].str.len()

    # token
    df["len token"] = df[text_column].apply(lambda x: len(x.split()))

    # stopwords
    stop = stopwords.words("english")
    df[text_column] = df[text_column].apply(
        lambda x: " ".join([word for word in x.split() if word not in (stop)])
    )

    df["len stop"] = df[text_column].apply(lambda x: len(x.split()))
    df["len stop unique"] = df[text_column].apply(lambda x: len(set(x.split())))
    logger.info("Stopwords removed")

    # stem
    stemmer = PorterStemmer()

    df[text_column] = df[text_column].apply(
        lambda x: " ".join([stemmer.stem(word) for word in x.split()])
    )

    df["len stem"] = df[text_column].apply(lambda x: len(x.split()))
    df["len stem unique"] = df[text_column].apply(lambda x: len(set(x.split())))
    logger.info("Dataset stemmed")

    return df.drop(text_column, axis=1)


def main(index: str):
    doc_path = config[index]["docs"]

    # calculate stats
    df = load_data(doc_path)
    df = clac_stats(df, "contents")

    # save stats
    df.to_csv("data/doc_{}_stats.csv".format(index), index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate corpus stats for a dataset.")

    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Name of the dataset in the config file",
    )

    args = parser.parse_args()

    main(args.index)
