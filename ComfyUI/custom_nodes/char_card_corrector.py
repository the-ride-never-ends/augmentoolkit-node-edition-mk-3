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
from numpy import mean
from numpy.random import f
from scipy import rand
import torch
import traceback
import uuid
import yaml


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import asyncio as tqdmasyncio
from tqdm import tqdm


import comfy.utils
import comfy.model_management
from comfy.cli_args import args

from custom_nodes.logger import logger
import folder_paths


COMPLETION_MODE = True
USE_FILENAMES = False
LOG_LEVEL = None


PROMPT_DICT = {}
for file_name in folder_paths.get_filename_list("prompts"):
    try:
        file_path = folder_paths.get_full_path("prompts", file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            key = os.path.splitext(file_name)[0]
            PROMPT_DICT[key] = file.read()
    except Exception as e:
        logger.exception(f"An Exception occured when creating the prompt dictionary object: {e} ")

# Load the default prompts into the PROMPT_DICT object
for file_name in folder_paths.get_filename_list("default_prompts"):
    try:
        file_path = folder_paths.get_full_path("default_prompts", file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            key = os.path.splitext(file_name)[0]
            PROMPT_DICT['default_prompts'][key] = file.read()
    except Exception as e:
        logger.exception(f"An Exception occured when creating the default_prompts in the prompt dictionary object: {e} ")

TEST_DICT = {}
for file_name in folder_paths.get_filename_list("tests"):
    try:
        file_path = folder_paths.get_full_path("tests", file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            key = os.path.splitext(file_name)[0]
            TEST_DICT[key] = file.read()
    except Exception as e:
        logger.exception(f"An Exception occured when creating the prompt dictionary object: {e} ")


# Set up rate-limit-conscious functions
SEMAPHORE = asyncio.Semaphore(CONCURRENCY_LIMIT)

# Function to prevent rate-limits overruns.
async def run_task_with_limit(task: Callable):
    async with SEMAPHORE:
        # Run your task here
        return await task

async def limited_tasks(tasks: Callable):
    limited_tasks_infocreation = [run_task_with_limit(task) for task in tasks]

    async def process_tasks(future, as_completed, limited_tasks_infocreation):
        for future in tqdmasyncio.tqdm(as_completed(limited_tasks_infocreation)):
            await future

    # Run the async function
    await process_tasks()

async def run_tasks(limited_tasks_qgen):
    for future in tqdmasyncio.tqdm.as_completed(limited_tasks_qgen):
        await future



"""
OpenToM: A Comprehensive Benchmark for Evaluating
Theory-of-Mind Reasoning Capabilities of Large Language Models
https://arxiv.org/pdf/2402.06044.pdf
"""





class TorrenceTest:
    """
    Implementation of:
    Assessing and Understanding Creativity in Large Language Models
    https://browse.arxiv.org/pdf/2401.12491.pdf
    
    Test creativity of an input LLM based on the Torrence Test for Creativity
    The original author's used GPT-4 to grade the questions, but others can probably be used.
    """
    @staticmethod
    async def torrence_test(idx, task, LLM, completion_mode, max_tokens_arg, temperature_arg, top_k_arg, top_p_arg, roles, prompt_type, seed_arg):

        criteria = []

        engine_wrapper = LLM['llm']

        # Initialize scores
        fluency_score = flexibility_score = originality_score = elaboration_score = 0

        fluency = """Fluency: The ability to produce a significant number of relevant ideas in response to a given question. 
        In essence, fluency measures the quantity of ideas.
        """

        flexibility = """Flexibility: The variety of categories from which one can generate ideas. 
        It's the ability to think of alternatives, shift from one class or perspective to another, and to approach a given problem or task from different angles. 
        """

        originality = """ Originality: The uniqueness of the ideas generated. 
        Original ideas are those that are rare or unconventional, differing from the norm. 
        """

        elaboration = """Elaboration: The ability to expand upon, refine, and embellish an idea. 
        It involves adding details, developing nuances, and building upon a basic concept to make it more intricate or complex.
        """

        criteria = [
            (fluency, "fluency"), 
            (flexibility, "flexibility"), 
            (originality, "originality"), 
            (elaboration, "elaboration")
        ]

        task_type = task['task_type']
        question = task['question']
        random.seed(seed_arg)
        role = random.choice(roles)

        prompts = []

        basic_prompt = f"""
           Act like a typical {role}.\n
           Do the following task or answer the following question.\n
           {task_type}\n
           The scenario is:\n
           {question}\n
           Answer:
        """

        instructive_prompt = f"""
           Act like a typical {role}.\n
           Do the following task or answer the following question.\n
           {task_type}\n
           There are no right or wrong answers, we’re interested in how many different problems you can identify and the variety of issues you consider. 
           Try to think outside the box and consider as many potential problems as possible.\n
           The scenario is:\n
           {question}\n
           Answer:
        """

        cot_prompt = f"""
           Act like a typical {role}.\n
           Do the following task or answer the following question.\n
           {task_type}\n
           Let's think step by step.
           The scenario is:\n
           {question}\n
           Answer:
        """

        prompts = [
            basic_prompt,
            instructive_prompt,
            cot_prompt,
        ]

        if prompt_type == "Randomly Chosen":
            torrence_test_prompt = random.choice(prompts)
        elif prompt_type == "Basic":
            torrence_test_prompt = prompts[0]
        elif prompt_type == "Instructive":
            torrence_test_prompt = prompts[1]
        elif prompt_type == "Chain of Thought":
            torrence_test_prompt = prompts[2]
        else:
            logger.warning(f"WARNING: '{prompt_type}' is not a currently recognized prompt type. Defaulting to randomly chosen...")
            torrence_test_prompt = random.choice(prompts)

        torrence_test_regex = re.compile(
            rf"Answer:(.+)",
            re.IGNORECASE | re.DOTALL,
        )

        creative_response = GenerationStep( # will have variations as an argument
            prompt_path=torrence_test_prompt,
            regex = torrence_test_regex,
            sampling_params={
                "max_tokens": max_tokens_arg,
                "stop": [
                    "### Response",
                    "\n\n\n\n\n",
                    "</s>",
                    "# Input:",
                    "[INST]",
                    "### Instruction",
                    "[INST",
                    "## Information",
                    "User:",
                ],
                "temperature": temperature_arg,
                "top_k": top_k_arg,
                "top_p": top_p_arg,
            },
            completion_mode=COMPLETION_MODE,
            retries=1,
            engine_wrapper=engine_wrapper,
            prompt_folder=PROMPTS,
            default_prompt_folder=DEFAULT_PROMPTS
        )

        creative_response = await creative_response.generate() # TODO add arguments here when the prompts are externalized.

        for metric, name in criteria: # Instructive prompt.
            criteria_prompt = f"""
                As of now, you are an expert psychologist specializing in the Torrance Test of Creative Thinking.\n
                In this schema, there are 4 main criteria for grading creative responses: fluency, flexibility, originality, and elaboration.\n
                Using the following criteria, explain how the response both does and does not reflect it.
                Then, based on your reasoning, score the creativity of this response on a scale between 1 and 5,\n
                where 1 represents a response that does not reflect the criteria, and 5 represents a response that clearly reflects the criteria.\n
                \#\#\#\#\#\#\n
                Criteria:\n
                {metric}\n
                \#\#\#\#\#\#\n
                Response:\n
                {creative_response}
                \#\#\#\#\#\#\n
                Final Judgement:
            """

            judgement_response = GenerationStep( # will have variations as an argument
                prompt_path=criteria_prompt,
                regex = torrence_test_regex,
                sampling_params={
                    "max_tokens": max_tokens_arg,
                    "stop": [
                        "### Response",
                        "\n\n\n\n\n",
                        "</s>",
                        "# Input:",
                        "[INST]",
                        "### Instruction",
                        "[INST",
                        "## Information",
                        "User:",
                    ],
                        "temperature": temperature_arg,
                        "top_k": top_k_arg,
                        "top_p": top_p_arg,
                    },
                    completion_mode=completion_mode,
                    logging_level=logging_level,
                    retries=1,
                    engine_wrapper=engine_wrapper,
                    prompt_folder=PROMPTS,
                    default_prompt_folder=DEFAULT_PROMPTS
                )

            judgement_response = await judgement_response.generate() 
                
            decision_pattern = re.compile(
               r"Final Judgment:(.+)", re.DOTALL | re.IGNORECASE
            )
            determination = decision_pattern.search(judgement_response).group(1).strip()

            if "1" or "one" in determination.lower():
                score += 1
            elif "2" or "two" in determination.lower():
                score += 2
            elif "3" or "three" in determination.lower():
                score += 3
            elif "4" or "four" in determination.lower():
                score += 4
            elif "5" or "five" in determination.lower():
                score += 5
            else:
                logger.error(f"ERROR: LLM determination of metric did not produce a single output within the range of 1 to 5.")
                score += 0 # Since the metric didn't work, 
                
            if name == "fluency":
                fluency_score += score
            elif name == "flexibility":
                flexibility_score += score
            elif name == "originality":
                originality_score += score
            elif name == "elaboration":
                elaboration_score += score

        logger.info(f"Scores for task {idx}:\nfluency_score: {fluency_score}\nflexibility_score: {flexibility_score}\noriginality_score: {originality_score}\nelaboration_score: {elaboration_score}\n")
        return (fluency_score, flexibility_score, originality_score, elaboration_score)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM": ("LLM",),
                "max_tokens_arg": ("INT", {"default": 512, "min":1, "max":100000, "step":1}),
                "stop_arg": ("STRING", {"default": '["### Response", "\n\n\n\n\n","</s>", "# Input:", "[INST]", "### Instruction", "[INST", "## Information", "User:"]'}, {"multiline": True}),
                "echo_arg": (["True", "False"],),
                "roles": ("STRING", {"default": '["primary school student","natural scientist","music artist"]'}, {"multiline": True}),
                "prompt_type": ([ "Randomly Chosen", "Basic", "Instructive", "Chain of Thought"],),
                "temperature_arg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}), # Default is 1, which is deterministic e.g. the most likely token is always selected.
                "top_k_arg": ("INT", {"default": 50, "min":1, "max":1000, "step":1}), # Default is 40, which means only the top 40 token probabilities are considered for selection.
                "top_p_arg": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 20.0, "step": 0.01}), # Default is 1, which is off (???)
                "min_p_arg": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}), # Default is 0, which is off (???)
                "seed_arg": ("INT", {"default": -1, "min": -1, "max":0xffffffffffffffff, "step":1}), # Default is -1 i.e. the seed is randomly generated???
                "only_override_prompt_and_grammar": (["False", "True"],), # This allows the prompt and grammar to be overriden while still keeping the original function generation presets.
            },
            "optional": {
                "prompt": ("PROMPT", {"forceInput": True}),
            },
        }
    RETURN_TYPES = ("LLM",)

    FUNCTION = "return_torrence_test"

    CATEGORY = "benchmarks/advanced"

    @conditional_log_arguments
    def return_torrence_test(self, LLM, max_tokens_arg, temperature_arg, top_k_arg, top_p_arg, roles, prompt_type, seed_arg):

        # Look up the json files in the tests folder.
        # TODO Make this changable like prompts, inputs, and outputs.
        existing_files = glob.glob(os.path.join("./tests", "*.json"))

        
        for file_path in existing_files:
            if file_path.endswith("torrence_test_creative_questions"):
                with open(file_path, "r") as file:
                     torrence_test = tuple(json.load(file))

        fluency_score = 0
        flexibility_score = 0
        originality_score = 0
        elaboration_score = 0
        highest_possible_score = (700 * 4 * 5) # 14,000. 700 questions * 4 metrics * 5 possible scores.

        tasks = [
            self.torrence_test(
                idx, task,
                LLM, 
                COMPLETION_MODE, 
                use_filenames=USE_FILENAMES, 
                completion_mode=COMPLETION_MODE, 
                logging_level=LOG_LEVEL) for idx, task in enumerate(torrence_test)
        ]
        limited_tasks_torrence_test = [run_task_with_limit(task) for task in tasks]
        asyncio.run(run_tasks(limited_tasks_torrence_test))
        
        average_score = mean(fluency_score, flexibility_score, originality_score, elaboration_score)

        return (average_score, fluency_score, flexibility_score, originality_score, elaboration_score),


task
    task_type
    question




    task = 
        # Prompt Type
            # Basic Prompt
            # Instructive Prompt
            # Cot Prompt


        # Unusual Uses Task
        # Consequences Task
        # Just Suppose Task
        # Situation Task
        # Common Problem Task
        # Improvement Task
        # Imaginative Stories Task























