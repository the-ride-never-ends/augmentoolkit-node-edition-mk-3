import ast
import asyncio
from email.utils import formataddr
import glob
import io
import importlib
import inspect
import itertools
import json
import logging
import os
import random
import re
#import sentiencepiece
import sys
import time
from numpy.random import f
from scipy import rand
import torch
import traceback
import uuid
import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

script_dir = os.path.dirname(os.path.realpath(__file__))
custom_nodes_path = os.path.join(script_dir, "ComfyUI", "custom_nodes")
sys.path.insert(0, custom_nodes_path)

from accelerate.utils import release_memory
from collections import Counter
from collections.abc import Callable, Awaitable
from datetime import datetime
from functools import partial, wraps
from llama_cpp import Llama, LlamaGrammar
from math import ceil
from tqdm import asyncio as tqdmasyncio
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.deberta.modeling_deberta import DebertaLayerNorm
from typing import Any, List, Tuple, Union

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

import comfy.utils
import comfy.model_management
from comfy.cli_args import args

from custom_nodes.logger import logger
import custom_nodes.augmentoolkit_async_functions
import custom_nodes.augmentoolkit_api_functions
from custom_nodes.grammars import Grammars

import folder_paths

# Try to import all packages for the Aphrodite and API nodes.
try:
    from aphrodite import (
        EngineArgs,
        AphroditeEngine,
        SamplingParams,
        AsyncAphrodite,
        AsyncEngineArgs,
    )
except:
    logger.info("Aphrodite not installed; stick to Llama CPP or API modes")
    APHRODITE_NOT_INSTALLED = True

try:
    import openai>=0.10.2
except:
    logger.info("OpenAI client not installed.")
    OPENAI_NOT_INSTALLED = True

try:
    import together>=0.1.5
except:
    logger.info("Together.ai client not  installed.")
    TOGETHER_NOT_INSTALLED = True






# TODO Redo how APIs are implemented, per ComfyUI-Llama and NodeGPT.
# TODO Figure out how to get scientific notation to work.
# TODO This can obviously be made to work with other models. Figure out how.
# TODO Unhardcode the outputfile name for the 'prepare_date_for_together' function.
# TODO Allow users to choose which OpenAI model to use. Also the temperature too.
# This should probably be several different nodes, but then again, so should a lot of these nodes...
class AutoFinetune:
    """
    Implementation of yoheinakajima's autofinetune in node format.

    "autofinetune is a Python-based tool that automates the generation of conversational datasets 
    and uses them for model fine-tuning. It's particularly useful for projects where custom conversation patterns 
    and responses are needed, such as creating an AI that can classify messages or respond in a specific manner."

    Requirements:
    openai>=0.10.2
    together>=0.1.5
    From: https://github.com/yoheinakajima/autofinetune/blob/main/autofinetune.py

    :param client:
    :param target_conversations: The total number of conversation entries you aim to generate for your dataset.
    :param conversation_batch_size: The number of conversation entries to generate in each batch.
    :param suffix: A unique identifier for the fine-tuned model on the Together AI platform.
    :param objective: The overarching goal of the model being fine-tuned (e.g., classifying messages as spam or not spam).
    :param user_input_rules: Guidelines for the types of user inputs the script should generate (e.g., random email titles).
    :param assistant_response_rules: The desired behavior for the assistant's responses (e.g., responding with "spam" or "not spam").
    :param finetune_model: The together.ai model to be fine-tuned.
    :param finetune_n_epochs: Number of epochs to train the model for.
    :param finetune_n_checkpoints:
    :param finetune_batch_size:
    :param finetune_learning_rate: Learning rate for model training.
    :param OPENAI_API_KEY: The OpenAI api key
    :param TOGETHER_API_KEY: The together.ai api key
    :return None: The output is a fine-tuned model on TogetherAI's playground, and a bunch of conversation jsons.
    """
    def __init__(self):
        self.type = "output"

    @staticmethod
    def generate_conversations(client, target_conversations, conversation_batch_size, 
                               suffix, objective, user_input_rules, assistant_response_rules, 
                               override_system_prompt=None, override_user_prompt=None
    ):
        conversations = []  # Initialize the main conversations list
        generated_count = 0  # Keep track of how many conversations have been generated

        # Check if the prompts are being overridden.
        if override_system_prompt or override_user_prompt:
            if override_system_prompt is not None and override_user_prompt is None: # Only override system prompt
                prompt_content = {
                    "conversation_batch_size": conversation_batch_size
                }
                system_prompt, _ = load_external_prompt_and_grammar("autofinetune_system_prompt", "dummy_grammar", prompt_content)

                input_messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"""Start generating the conversation array.
                        Purpose of the conversation data is to fine tune a model.
                        The obective of the model we are fine-tuning is: {objective}.
                        User input rules: {user_input_rules}.
                        Assistant response rules: {assistant_response_rules}.
                        """
                    }
                ]
            elif override_system_prompt is None and override_user_prompt is not None: # Only override user prompt
                prompt_content = {
                    "objective": objective,
                    "user_input_rules": user_input_rules,
                    "assistant_response_rules": assistant_response_rules
                }
                user_prompt, _ = load_external_prompt_and_grammar("autofinetune_user_prompt", "dummy_grammar", prompt_content)

                input_messages = [
                    {
                        "role": "system",
                        "content": f"""You are a helpful assistant designed to output conversations in JSON array format. Generate an array of {conversation_batch_size} elements, each being a conversation entry with a user input and your response, with the keys 'user' and 'assistant', one message each per conversation, with {conversation_batch_size} of these conversations in the array you produce.The purpose of this is to fine-tune a model based for the user using this data, so tailor the questions and answers to match the needs based on the user's input. The array should start with conversations as the top level with an array of conversations.
                        """
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            else: # Override both system and user prompt.
                system_prompt_content = {
                    "conversation_batch_size": conversation_batch_size
                }
                system_prompt, _ = load_external_prompt_and_grammar("autofinetune_system_prompt", "dummy_grammar", prompt_content)

                user_prompt_content = {
                    "objective": objective,
                    "user_input_rules": user_input_rules,
                    "assistant_response_rules": assistant_response_rules
                }
                user_prompt, _ = load_external_prompt_and_grammar("autofinetune_user_prompt", "dummy_grammar", prompt_content)

                input_messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
        else: # Default to the hardcoded prompts if the overrides aren't there.
            input_messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant designed to output conversations in JSON array format. Generate an array of {conversation_batch_size} elements, each being a conversation entry with a user input and your response, with the keys 'user' and 'assistant', one message each per conversation, with {conversation_batch_size} of these conversations in the array you produce.The purpose of this is to fine-tune a model based for the user using this data, so tailor the questions and answers to match the needs based on the user's input. The array should start with conversations as the top level with an array of conversations.
                    """
                },
                {
                    "role": "user",
                    "content": f"""Start generating the conversation array.
                    Purpose of the conversation data is to fine tune a model.
                    The objective of the model we are fine-tuning is: {objective}.
                    User input rules: {user_input_rules}.
                    Assistant response rules: {assistant_response_rules}.
                    """
                }
            ]

        # Generate completions until we reach the conversation target number.
        # TODO Un-hardcode this.
        while generated_count < target_conversations:
            # Generate the completion.
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                temperature = 1,
                response_format={"type": "json_object"},  # Enable JSON mode
                messages=input_messages
            )

            # Extract the response.
            generated_content = response.choices[0].message.content
            if DEBUG_MODE:
                logger.info(f"\n *** Completion from 'generate_conversations' function from class AutoFinetune *** \n{generated_content}\n*** Completion from 'generate_conversations' function from class AutoFinetune ***")

            # Save the response to a json file.
            try:
               generated_content = json.loads(response.choices[0].message.content)
               if "conversations" in generated_content:
                   for conv in generated_content["conversations"]:
                       if "user" in conv and "assistant" in conv:
                           conversations.append(conv)
                           generated_count += 1 
                       else:
                           logger.info("Skipping a conversation entry due to missing 'user' or 'assistant' key.")

            except json.JSONDecodeError:
                logger.error("ERROR in 'generate_conversations' function in class AutoFinetune: Error decoding JSON from response")

        logger.info(f"Generated {generated_count} conversations so far.")

        return json.dumps({"conversations": conversations}) 

    @staticmethod
    def prepare_date_for_together(generated_content):
        # Load the synthetic data.
        conversations_json = json.loads(generated_content)
        # Create a list to put it in.
        together_data = []
        # Reformat the synthetic data into user/assistant conversations.
        for conv in conversations_json["conversations"]:

            formatted_entry = {
                "text": f'<s>[INST] {conv["user"]} [/INST] {conv["assistant"]} </s>'
            }
            together_data.append(formatted_entry)

        # Save the data to the "conversations_dataset.jsonl"
        with open("conversations_dataset.jsonl", 'w') as outfile:
            for item in together_data:
                outfile.write(json.dumps(item) + '\n') # Save to a JSONL file
  
        return "conversations_dataset.jsonl"
        
    @staticmethod
    def upload_and_fine_tune(file_name, finetune_model, finetune_n_epochs, finetune_n_checkpoints, finetune_batch_size, finetune_learning_rate, suffix):
        file_id = together.Files.upload(file=file_name)["id"]

        fine_tune_response = together.Finetune.create(
            training_file=file_id,
            model=finetune_model,
            n_epochs=finetune_n_epochs,
            n_checkpoints=finetune_n_checkpoints,
            batch_size=finetune_batch_size,
            learning_rate=finetune_learning_rate,
            suffix=suffix,
            # wandb_api_key='YOUR_WANDB_API_KEY',
        )
        logger.info(f"Fine-tuning started:\n{fine_tune_response}")
        logger.info("Check your results/status at https://api.together.xyz/playground/jobs")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_conversations": ("INT", {"default": 100, "min":1, "max":100000, "step":1}),
                "conversation_batch_size": ("INT", {"default": 5, "min":1, "max":100000, "step":1}),
                "suffix": ("STRING", {"default": 'autofinetune'}),
                "objective": ("STRING", {"default": 'to generate a model that labels messages as spam or not spam.', "multiline": True}),
                "user_input_rules": ("STRING", {"default": 'Make it a random title of an email.', "multiline": True}),
                "assistant_response_rules": ("STRING", {"default": 'Only responds with "spam" or "not spam".',"multiline": True}),
                "finetune_model": ("STRING", {"default": 'togethercomputer/llama-2-7b-chat'}),
                "finetune_n_epochs": ("INT", {"default": 5, "min":1, "max":100000, "step":1}),
                "finetune_n_checkpoints": ("INT", {"default": 5, "min":1, "max":100000, "step":1}),
                "finetune_batch_size": ("INT", {"default": 4, "min":1, "max":1000, "step":1}),
                "finetune_learning_rate": ("FLOAT", {"default": 0.00001, "min": 0.000000001, "max": 1.0000000000, "step": 0.000000001}),
                "OPENAI_API_KEY": ("OPENAI_API_KEY",),
                "TOGETHER_API_KEY": ("TOGETHER_API_KEY",),
            },
            "optional": {
                "override_system_prompt": ("PROMPT", {"forceInput": True}),
                "override_user_prompt": ("PROMPT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "autofinetune"

    OUTPUT_NODE = True

    CATEGORY = "augmentoolkit_functions/advanced/debug"

    def autofinetune(self, target_conversations, conversation_batch_size, suffix, objective, user_input_rules, 
                     assistant_response_rules, finetune_model, finetune_n_epochs, finetune_n_checkpoints,
                     finetune_batch_size, finetune_learning_rate, OPENAI_API_KEY, TOGETHER_API_KEY,
                     override_system_prompt=None, override_user_prompt=None
    ):
        OpenAI.apikey = os.environ['OPENAI_API_KEY']
        together.api_key = os.environ['TOGETHER_API_KEY']
        client = OpenAI()

        logger.info("Autofinetune: Generating conversation data...")
        generated_content = self.generate_conversations(client, target_conversations, conversation_batch_size, suffix, objective, user_input_rules, assistant_response_rules, override_system_prompt=None, override_user_prompt=None)

        logger.info("Autofinetune: Preparing data for Together.ai...")
        file_name = self.prepare_data_for_together(generated_content)

        logger.info("Autofinetune: Uploading and starting fine-tuning...")
        self.upload_and_fine_tune(file_name, finetune_model, finetune_n_epochs, finetune_n_checkpoints, finetune_batch_size, finetune_learning_rate, suffix)

        return None












