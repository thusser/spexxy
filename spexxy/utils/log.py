import logging

from ..version import version


def setup_log(name: str, filename: str = None, stream: bool = True, mode: str = 'w',
              header: bool = True) -> logging.Logger:
    """Sets up a new logger object

    Args:
        name: Name of new logger.
        filename: If given, name of file to write logs to.
        stream: If True, also log to stdout.
        mode: Open mode for log file (w/a).
        header: If True, immediately writes a spexxy header into the log.

    Returns:
        The new logger object
    """

    # get logger
    log = logging.getLogger(name)

    # remove existing handlers
    shutdown_log(name)

    # log format
    log_formatter = logging.Formatter("%(asctime)s[%(levelname)-8s]: %(message)s", datefmt='%m/%d/%Y %H:%M:%S')

    # create file handler
    if filename is not None:
        file_handler = logging.FileHandler(filename, mode=mode)
        file_handler.setFormatter(log_formatter)
        log.addHandler(file_handler)

    # stream handler as well?
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        log.addHandler(stream_handler)

    # log header
    if header:
        log.info(r"")
        log.info(r" ___ _ __   _____  ____  ___   _ ")
        log.info(r"/ __| '_ \ / _ \ \/ /\ \/ / | | |")
        log.info(r"\__ \ |_) |  __/>  <  >  <| |_| |")
        log.info(r"|___/ .__/ \___/_/\_\/_/\_\\__, |")
        log.info(r"    | |                     __/ |")
        log.info(r"    |_|        v%3s        |___/ " % version())
        log.info(r"")
        log.info(r"        Tim-Oliver Husser        ")
        log.info(r"    thusser@uni-goettingen.de    ")
        log.info(r"")

    # return it
    return log


def shutdown_log(name: str):
    """Shuts down a logger objects, especially removes all handlers, so that we don't get a
    'too many open files' error.

    Args:
        name: Name of logger to shut down.
    """

    # get logger
    log = logging.getLogger(name)

    # remove existing handlers
    for handler in log.handlers[:]:
        log.removeHandler(handler)
