import logging

import tqdm  # type: ignore


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


def get_console_handler() -> logging.Handler:
    """Create a console handler for logging.

    Returns:
        logging.Handler: The console handler.
    """
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    return console_handler


def get_new_logger(name: str) -> logging.Logger:
    """Get the logger from a given name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(name)

    ch = get_console_handler()
    logger.addHandler(ch)
    return logger


logger_tqdm = get_new_logger("tqdm")
logger_tqdm.addHandler(TqdmLoggingHandler())

logger = get_new_logger(__name__)
