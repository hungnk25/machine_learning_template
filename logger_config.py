# This code section should run only once at the start of the program
# to setup the logger
# This script must be imported before any logger call

import logging
import logging.config
import os
import re
import sys
from pathlib import Path


def create_directory(target_directory):
    if os.path.exists(target_directory):
        for the_file in os.listdir(target_directory):
            file_path = os.path.join(target_directory, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    else:
        os.makedirs(target_directory)


# create the log folder, which is hard-coded
log_dir = os.path.join(Path(__file__).parent, "logs")
print(f"Create directory for logs: {log_dir}")
create_directory(log_dir)

logger = logging.getLogger("main_logger")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(os.path.join(log_dir, "main_logger.log"))

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

format = logging.Formatter(
    "%(asctime)s - %(name)s-%(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(format)
console.setFormatter(format)
logger.addHandler(handler)
logger.addHandler(console)

