import sys

from loguru import logger
from loguru._logger import Logger

from zeroband.training.config import LogConfig
from zeroband.training.world import World
from zeroband.utils.logger import get_logger, set_logger

NO_BOLD = "\033[22m"
RESET = "\033[0m"


def setup_logger(log_config: LogConfig, world: World) -> Logger:
    if get_logger() is not None:
        raise RuntimeError("Logger already setup. Call reset_logger first.")

    # Define the time format for the logger.
    time = "<dim>{time:HH:mm:ss}</dim>"
    if log_config.utc:
        time = "<dim>{time:zz HH:mm:ss!UTC}</dim>"

    # Define the colorized log level and message
    message = "".join(
        [
            " <level>{level: >7}</level>",
            f" <level>{NO_BOLD}",
            "{message}",
            f"{RESET}</level>",
        ]
    )

    # Add parallel information to the format
    if world.world_size > 1:
        format = time + f"[ Rank {world.rank} ]" + message
    else:
        format = time + message
    if log_config.level.upper() == "DEBUG":
        format += "".join([f"<level>{NO_BOLD}", " [{file}::{line}]", f"{RESET}</level>"])

    # Remove all default handlers
    logger.remove()

    # Install new handler on all ranks, if specified. Otherwise, only install on the main rank
    if log_config.all_ranks or world.rank == 0:
        logger.add(
            sys.stdout, format=format, level=log_config.level.upper(), enqueue=True, backtrace=True, diagnose=True
        )

    # Disable critical logging
    logger.critical = lambda _: None

    # Bind the logger to access the rank
    set_logger(logger)

    return logger
