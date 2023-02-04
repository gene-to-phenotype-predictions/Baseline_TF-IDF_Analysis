import logging
import sys

logging.captureWarnings(True)

debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

for handler in list(logger.handlers):
    logger.removeHandler(handler)

streamHandler = logging.StreamHandler(sys.stdout)
streamHandler.setLevel(logging.INFO)
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

fileHandler = logging.FileHandler('debug.log', mode='a', encoding=None, delay=False, errors=None)
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(debug_formatter)
logger.addHandler(fileHandler)