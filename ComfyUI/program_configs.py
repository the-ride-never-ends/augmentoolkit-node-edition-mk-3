import os
import yaml
import logging
from logger import logger


# Define base paths and config path
base_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(base_path, "config.yaml")


program_defaults = {
    "DEBUG_MODE": True, # Debug global. True by default
    "API_KEY": "", # Currently set to nothing.
    "ASSISTANT_MODE": True,  # change to true if you want all conversations to be with an "AI language model" and not characters. 
                       # Useful for more professional use cases.
    "BASE_URL": "https://api.mistral.ai/v1/",
    "COMPLETION_MODE": False,
    "CONCURRENCY_LIMIT": 90,
    "DOUBLE_CHECK_COUNTER": 3,   # Set to 1 to check outputs only once; set to 2 to check twice; set to 3 to check thrice, etc. 
                                 # Set to 0 to break everything in vet_question_loop() and elsewhere. 
                                 # Set to -1 and cause the universe to implode?
    "GRAPH": False,
    "N_CHARACTERS_SAME": 3, # Set to 1 to check outputs only once; set to 2 to check twice; set to 3 to check thrice, etc. 
                            # Set to 0 to break everything in vet_question_loop() and elsewhere. 
                            # Set to -1 and cause the universe to implode?
    "REARRANGEMENTS_TO_TAKE": 3,  # How many of the possible permutations of tuples in a group to take and make multiturn convs out of. 
                            # Adjust higher to get more data out of less text, but it might be a bit repetitive. 
                            # NOTE your eval loss will be basically worthless if you aren't careful with how you shuffle your dataset when you're about to train.
    "TEXT_MANUALLY_CLEANED": False, # If you've manually cut out all the parts of the text not worthy for questions, you can skip the first LLM step. 
                               # NOTE I might actually recommend doing this if your source text is small, given how permissive the filtering prompt is; 
                               # it really only disallows metadata.
    "USE_FILENAMES": False,
    "USE_SUBSET": True,
    # Path globals.
    "INPUT": os.path.join(base_path, "input"), 
    "OUTPUT": os.path.join(base_path, "output"),
    "PROMPTS": os.path.join(base_path, "prompts"),
    "DEFAULT_PROMPTS": os.path.join(base_path, "prompts"),
    "LOG_LEVEL": "Debug"
}


def load_defaults_config(file_path: str):

    with open(file_path, 'r') as file:
        obj_conf = yaml.safe_load(file)

        # Apply configs from YAML file
        for key, value in obj_conf.items():
            try: 
                # Check if path modification is necessary
                if key in ["INPUT", "OUTPUT", "DEFAULT_PROMPTS", "PROMPTS"] and os.name == "nt":
                    value = value.replace("/", os.sep)
                # Turn the configs into global variables.
                globals()[key] = value 
                print(f"Global variable '{key}' set to '{value}'")
            except Exception as e:
                print(f"WARNING: Could not find global variable '{key}' in 'config.yaml'. Defaulting to hardcoded global variable preset.")


# Load the config file contents.
try:
    print("Loading global variable presets from 'config.yaml'...")
    load_defaults_config(f'{config_path}')
    print("'config.yaml' loaded.")
except Exception as e:
    logger.exception(f"WARNING: Could not load 'config.yaml' file due to: {e}\n Defaulting to hardcoded global variable presets.\n")


def get_config(name: str):
    if name not in globals():
        logger.warning(f"WARNING: Global variable '{name}' not found. Searching program defaults...")
        try:
            return globals().get(name, program_defaults[name])
        except:
            logger.error(f"ERROR: Global variable '{name}' not found.")
            raise ValueError(f"Global variable '{name}' not found.")
    return globals()[name]


# Sample config yaml for augmentoolkit
"""
PATH:
  INPUT: "./input"
  OUTPUT: "./output"
  DEFAULT_PROMPTS: "./prompts" # the baseline prompt folder that Augmentoolkit falls back to if it can't find a step in the PROMPTS path
  PROMPTS: "./prompts" # Where Augmentoolkit first looks for prompts
API:
  API_KEY: ""
  BASE_URL: "https://api.mistral.ai/v1/"
  LOGICAL_MODEL: "mistral-large" #Not used in Comfy, as you can select the model in the UI directly.
  LARGE_LOGICAL_MODEL: "mistral-large" #Not used in Comfy, as you can select the model in the UI directly.
SYSTEM:
  USE_FILENAMES: False
  ASSISTANT_MODE: False 
  DEBUG_MODE: True
  DOUBLE_CHECK_COUNTER: 3
  USE_SUBSET: True
  REARRANGEMENTS_TO_TAKE: 3
  CONCURRENCY_LIMIT: 90
  COMPLETION_MODE: False
  MODE: "api" # can be one of "api"|"aphrodite"|"llamacpp" Currently redundant due to mode being chosen implicitly by which loader node is used.
  GRAPH: False
  
ASSISTANT_MODE = obj_conf['']['']
DEBUG_MODE = obj_conf['']['']
DOUBLE_CHECK_COUNTER = obj_conf['']['']
REARRANGEMENTS_TO_TAKE = obj_conf['']['']
TEXT_MANUALLY_CLEANED = obj_conf['']['']
"""
