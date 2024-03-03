import asyncio
import glob
from importlib.machinery import DEBUG_BYTECODE_SUFFIXES
import inspect
import itertools
import json
import logging
import os
import random
import re
import string
import time
import traceback
import uuid

from tqdm import asyncio as tqdmasyncio
from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple
from math import ceil
from aphrodite import SamplingParams

from custom_nodes import logger, augmentoolkit
import folder_paths

from custom_nodes.logger import logger

from custom_nodes.augmentoolkit import (
    API_KEY,
    ASSISTANT_MODE, # Global variables
    BASE_URL,
    COMPLETION_MODE,
    CONCURRENCY_LIMIT, 
    DEBUG_MODE, 
    DOUBLE_CHECK_COUNTER, 
    GRAMMAR_DICT,
    GRAPH,
    MODE,
    NAMES,
    PROMPT_DICT,
    REARRANGEMENTS_TO_TAKE,
    USE_FILENAMES, 
    USE_SUBSET, 
    EngineWrapper, # Functions
    extract_name, 
    format_external_text_like_f_string,
    format_qatuples,
    limited_tasks,
    load_external_prompt_and_grammar,
    override_prompt_and_grammar,
    run_task_with_limit,
    special_instructions,
    strip_steps,
    write_output_to_file,
)



class ChatGPT:
    """
    Load a model from OpenAI via its API.
    Modified from: https://github.com/xXAdonesXx/NodeGPT/blob/main/API_Nodes/ChatGPT.py
    :param model_name:
    :param model_name:
    :return LLM:
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "gpt-4"})
            },
            "optional": {
                "open_ai_api_key": ("STRING", {"default": None}),
                "open_ai_api_base_url": ("STRING", {"default": None}),
            }
        }

    RETURN_TYPES = ("LLM",)
    FUNCTION = "execute"
    CATEGORY = "loaders/api"

    def execute(self, model_name, open_ai_api_key, open_ai_api_base_url):
        # Load the api key from the 'config.yaml' file if a custom key is not present
        if open_ai_api_key is None:
            logger.info("'open_ai_api_key' argument not specified. Defaulting to API key from config.yaml.")
            try:
                engine_wrapper = EngineWrapper(model=model_name, mode="api", api_key=API_KEY, base_url=BASE_URL,)
                config_list = [
                    {
                        'llm': engine_wrapper,
                        'type': 'api',
                        'api_subtype': 'openai',
                        'api_key': API_KEY,
                    }
                ]
            except Exception as e:
                logger.exception(f"An Exception occured when trying to import the OpenAI API key: {e}")
                print("This may have occured because the API_KEY or BASE_URL in the 'config.yaml' file does not exist or is for another service.")
                raise e
        else:
            try:
                engine_wrapper = EngineWrapper(model=model_name, mode="api", api_key=open_ai_api_key, base_url=open_ai_api_base_url,)
                config_list = [
                    {
                        'llm': engine_wrapper,
                        'type': 'api',
                        'api_sub_type': 'openai',
                        'api_key': open_ai_api_key,
                    }
                ]
            except Exception as e:
                logger.exception(f"An Exception occured in 'execute' function in class 'ChatGPT' while trying to import the specified OpenAI API key: {e}")
                raise e

        return ({"LLM": config_list},)


class LM_Studio:
    """
    Load a model from LM Studio via its API.
    Modified from: https://github.com/xXAdonesXx/NodeGPT/blob/main/API_Nodes/ChatGPT.py
    :param model_name:
    :param model_name:
    :return LLM:
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": None}),
                "api_key": ("STRING", {"default": "NULL"}),
                "api_base": ("STRING", {"default": "http://localhost:1234/v1"}),
            }
        }

    RETURN_TYPES = ("LLM",)
    FUNCTION = "execute"
    CATEGORY = "loaders/api"

    def execute(self, model_name, api_key, api_base):
        try:
            engine_wrapper = EngineWrapper(model=model_name, mode="api", api_key=api_key, base_url=api_base,)
            config_list = [
                {
                    'llm': engine_wrapper,
                    'type': 'api',
                    'api_subtype': 'lm_studio',
                    'api_key': api_key,
                    'base_url': api_base,
                }
            ]
        except Exception as e:
            logger.exception(f"An Exception occured in 'execute' function in class 'LM_Studio' : {e}")
            raise e

        return ({"LLM": config_list},)


# TODO figure out what api_type means.
# TODO Write function documentation.
class Ollama:
    """
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "ollama/mistral"}),
                "api_type": ("STRING", {"default": "litellm"}),
                "api_base": ("STRING", {"default": "http://0.0.0.0:8000"})
            }
        }

    RETURN_TYPES = ("LLM",)
    FUNCTION = "execute"
    CATEGORY = "loaders/api"

    def execute(self, model_name, api_type, api_base):
        try:
            engine_wrapper = EngineWrapper(model=model_name, mode="api", base_url=api_base,)
            config_list = [
                {
                    'model': engine_wrapper,
                    'type': 'api', 
                    'apit_subtype': 'ollama',
                    'api_type': api_type, #Holdover from the original code. Kept for backwards compatability, if it's even feasible.
                    'api_base': api_base,
                }
            ]
        except Exception as e:
            logger.exception(f"An Exception occured in 'execute' function in class 'Ollama' : {e}")
            raise e

        return ({"LLM": config_list},)


class Mistral:
    pass


class KobaldCpp:
    pass


NODE_CLASS_MAPPINGS = {
    "ChatGPT": ChatGPT,
    "KobaldCpp": KobaldCpp,
    "Mistral": Mistral,
    "LM_Studio": LM_Studio,
    "Ollama": Ollama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatGPT": "Load Model (OpenAI)",
    "KobaldCpp": "Load Model (Kobald CPP)",
    "Mistral": "Load Model (Mistral)",
    "LM_Studio": "Load Model (LM Studio)",
    "Ollama": "Load Model (Ollama)",
}






















