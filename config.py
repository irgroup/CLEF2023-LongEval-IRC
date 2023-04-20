import logging
import tqdm

CONFIG = {
    "t1": {
        "index_name": "index_t1",
        "docs": "data/publish/English/Documents/Trec/",
        "topics": "data/publish/English/Queries/train.trec",
        "qrels": "data/publish/French/Qrels/train.txt",
    }
}

INDEX_DIR = "./data/index/"


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class LETORLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET) -> None:
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(TqdmLoggingHandler())
logger.addHandler(LETORLoggingHandler())
