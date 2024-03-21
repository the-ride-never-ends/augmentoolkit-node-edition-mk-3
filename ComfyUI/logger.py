import os
import logging
from datetime import datetime

##################################
#### SETTING UP DEBUG LOGGING ####
##################################

# Brought to you by ChatGPT

# Folder for log files
base_path = os.path.dirname(os.path.realpath(__file__))
log_folder = os.path.join(base_path, "debug_logs")

# Create the folder if it doesn't exist
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Create a logger
logger = logging.getLogger("augmentoolkit_logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Check if the logger already has handlers
if not logger.handlers:
    # Get current date and time for the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(log_folder, f"debug_log_{current_time}.log")

    # Create handlers (file and console)
    file_handler = logging.FileHandler(filename)
    console_handler = logging.StreamHandler()

    # Set level for handlers
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
