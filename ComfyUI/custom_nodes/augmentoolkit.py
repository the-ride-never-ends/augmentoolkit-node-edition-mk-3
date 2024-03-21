import asyncio
import glob
import io
import inspect
import itertools
import json
import logging
import os
import random
import re
import time
#import sentiencepiece
import string
import sys
import uuid

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

script_dir = os.path.dirname(os.path.realpath(__file__))
custom_nodes_path = os.path.join(script_dir, ".." "custom_nodes")
sys.path.insert(0, custom_nodes_path)

from accelerate.utils import release_memory
from collections import Counter
from collections.abc import Callable
from functools import partial, wraps
from llama_cpp import Llama, LlamaGrammar
from math import ceil
from numpy.random import f, rand
from tqdm import asyncio as tqdmasyncio
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Any, List, Tuple, Union

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

from comfy.cli_args import args
import custom_nodes.augmentoolkit_async_2_functions as aug_async

script_dir = os.path.dirname(os.path.realpath(__file__))
custom_nodes_path = os.path.join(script_dir, "..", "custom_nodes")
sys.path.insert(0, custom_nodes_path)

import folder_paths
from logger import logger
from program_configs import get_config
from engine import EngineWrapper, format_external_text_like_f_string, load_external_prompt_and_grammar

# TODO Change to "augmentoolkit_async_functions" when testing is complete.

# Try to import all packages for the Aphrodite and API nodes.
if os.name == "nt":
    print("Note: Aphrodite-engine currently does not support Windows.\n Use Linux or WSL to run augmentoolkit with aphrodite-engine.\nOnly Llama CPP or API modes will run.")
    APHRODITE_NOT_INSTALLED = True

if os.name == "posix":
    try:
        print("augmentoolkit.py attempting to import Aphrodite-engine...")
        from aphrodite import (
            SamplingParams,
        )
        APHRODITE_NOT_INSTALLED = False
        print("Success!")
    except:
        print("Aphrodite-engine not installed. Only Llama CPP or API modes will run.")
        APHRODITE_NOT_INSTALLED = True

#########################################
############ AUGMENTOOLKIT ##############
#########################################


############### NOTES ###################
# All node classes are in alphabetical order.
# Back-end functions that many nodes call upon are organized alphabetically in the section "helper functions".
# Back-end functions that only specific nodes call upon are organized alphabetically under the class node they go to in the section "node-specific functions".
#########################################

#TODO Finish refactoring LLM nodes so that they all can accept llama-cpp setting, prompt, and grammar overrides.
#TODO Make these functions async so that this can run aphrodite.
#TODO Linux in general. MacOS eventually.

################################################################
#### DIRECTORIES, EXTENSIONS, AND GLOBAL VARIABLE FUNCTIONS ####
################################################################

# These do nothing at the moment. May change later - KR
SOURCE_TEXTS = [
    "Simple Sabotage, by the Office of Strategic Services, published 1944.txt",
    "Principles of Chemistry, by Demitry Mendeleev, published 1897.txt",
]

NAMES = [ # Replaces "Albert" in scenarios. Needs to be western male names to avoid pronoun and setting inconsistencies).
    "William",
    "James",
    "John",
    "Robert",
    "Michael",
    "Charles",
    "George",
    "Joseph",
    "Edward",
    "Henry",
    "Thomas",
    "David",
    "Richard",
    "Daniel",
    "Matthew",
    "Alexander",
    "Benjamin",
    "Christopher",
    "Nicholas",
    "Samuel",
]

#########################################
######### ASYNC HELPER FUNCTIONS ########
#########################################

# Set up rate-limit-conscious functions
SEMAPHORE = asyncio.Semaphore(get_config("CONCURRENCY_LIMIT"))

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

#########################################
########### HELPER FUNCTIONS ############
#########################################

# Decorator function to log a function's arguments.
# Useful for recording LLM presets.
def log_arguments(func: Callable):
    def wrapper(*args, **kwargs):
        if get_config("DEBUG_MODE"):
            # Get the names and input values of the function's arguments.
            func_args_names = inspect.signature(func).parameters.keys()
            args_names = list(func_args_names)[:len(args)]
            args_dict = dict(zip(args_names, args))

            # Merge args_dict with kwargs to have a complete map of argument names to their values
            all_args = {**args_dict, **kwargs}
            
            # Log function name and the arguments it received with their names
            logging.debug(f"Called {func.__name__} with arguments: {all_args}")
        return func(*args, **kwargs)
    return wrapper


# Conditional decorator function for log arguments.
# Allows it be controlled via DEBUG_MODE global variable.
def conditional_log_arguments(func: Callable):
    if get_config("DEBUG_MODE"):
        return log_arguments(func)
    else:
        return func


def extract_capital_letters(input_string):
    capital_letters = []
    for char in input_string:
        if char.isupper():
            capital_letters.append(char)
    return capital_letters


def select_random_capital(exclusions):
    # Create a list of capital letters excluding the ones in the exclusions list
    capitals = [letter for letter in string.ascii_uppercase if letter not in exclusions]

    # Select a random capital letter from the filtered list
    if capitals:
        return random.choice(capitals)
    else:
        return "No available capital letters to choose from"


def extract_first_words(character_name: str, text: str):
    # Regular expression pattern to extract first word after the character's name
    pattern = fr"{character_name}: \"(\w+)"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    return matches


def extract_name(str: str):
    # Regular expression to match 'Name:' followed by any characters until the end of the line
    name_regex = r"^Name:\s*([^\s]*)"

    # Searching in the multiline string
    match = re.search(name_regex, str, re.MULTILINE)

    if match:
        name = match.group(1)
        logger.info(f"Extracted name: {name}")
        return name
    else:
        logger.info("No name found, retrying with different regex")
        name_regex = r"Name: *([^\\]*)"

        # Searching in the multiline string
        match = re.search(name_regex, str, re.MULTILINE)

        if match:
            name = match.group(1)
            logger.info(f"Extracted name: {name}")
            return name


# For the reword step (ONLY USE IF JUDGEMENT IS REWORD, OTHERWISE WE JUST IGNORE THE LAST BIT OF THE GEN)
def extract_question_answer(response: str):
    # Define the regex pattern to match the question and answer
    pattern = r"### Question Rewording \(using text details as reference\):\nQuestion: (.+?)\nAnswer: (.+?)\n"

    # Search for the pattern in the response
    match = re.search(pattern, response)

    # Extract and return the question and answer if a match is found
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        return question, answer
    else:
        logger.info("Returned none, failed to match")
        return None, None


def extract_steps(text: str, steps=[2, 4, 5]):
    """
    Extracts the specified steps from the text.

    Args:
    text (str): The input text containing various steps.
    steps (list of int): The step numbers to extract.

    Returns:
    str: A new string with each specified step's content on its own line.
    """
    step_pattern = "|".join([f"Step {step}\." for step in steps])
    matches = re.findall(
        f"({step_pattern})\s*(.*?)\s*(?=(Step \d\.|$))", text, re.DOTALL
    )

    # Extract and join the matched content, skipping the "Step n." part
    extracted_text = "\n".join(match[1].strip() for match in matches)
    return extracted_text


def format_qatuples(qatuples: Tuple):
    strlst = []
    for qatuple in qatuples:
        strlst.append(
            f"""Question: \"\"\"{qatuple[0]}\"\"\"
Answer: \"\"\"{qatuple[1]}\"\"\""""
        )
    return "\n\n".join(strlst)


def format_qatuples_noquotes(qatuples: Tuple):
    strlst = []
    for idx, qatuple in enumerate(qatuples):
        strlst.append(f"""{idx + 1}. {qatuple[0]}""")
    return "\n".join(strlst)


def escape_unescaped_quotes(s):
    # Initialize a new string to store the result
    result = ""
    # Iterate through the string, keeping track of whether the current character is preceded by a backslash
    i = 0
    while i < len(s):
        # If the current character is a quote
        if s[i] == '"':
            # Check if it's the first character or if the preceding character is not a backslash
            if i == 0 or s[i-1] != '\\':
                # Add an escaped quote to the result
                result += r'\"'
            else:
                # Add the quote as is, because it's already escaped
                result += '"'
        else:
            # Add the current character to the result
            result += s[i]
        i += 1
    return result

def make_id() -> str:
    return str(uuid.uuid4())

# Note: This function has been largely depricated due to redundancy.
"""
def override_prompt_and_grammar(function_name: str, override_prompt=None, override_grammar=None, prompt_content=None) -> Union[str, str]:

    # Make sure the function name is a string, if it isn't already one.
    function_name = str(function_name)

    # I try so hard!
    try:
        # Override prompt only.
        if override_prompt is not None and override_grammar is None:
            try:
                print(f"Overriding prompt for {function_name} function...")
                # Load the override prompt.
                prompt = format_external_text_like_f_string(override_prompt, prompt_content)
                print(f"Prompt override for '{function_name}' function loaded successfully.")

                # Load the regular grammar.
                # Redirect stdout and stderr to suppress output from LlamaGrammar.from_string()
                # Otherwise, it prints the entire grammar to the console and makes the debug logs a lot longer than they need to be.
                print(f"Loading grammar for '{function_name}' function...")
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()

                # Load the regular grammar.
                grammar = LlamaGrammar.from_string(fr"{GRAMMAR_DICT[f'{function_name}_grammar']}")

                # Restore stdout and stderr after the grammar is loaded.
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                print(f"Grammar for '{function_name}' function loaded successfully.")

            except:
                logger.exception(f"An Exception occurred in '{function_name}' function while trying to override its prompt: {e}")
                print(f"Check the override prompt file. The override prompt's text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

        # Override grammar only.
        elif override_prompt is None and override_grammar is not None:
            try:
                # Load the regular prompt
                print(f"Loading prompt for '{function_name}' function...")
                prompt = format_external_text_like_f_string(PROMPT_DICT[f'{function_name}'], prompt_content)

                # Load the override grammar.
                # We don't supress the output for override grammar, since it's useful to see a printout of it during debugging.
                print(f"Overriding prompt for {function_name} function...")
                grammar = LlamaGrammar.from_string(fr"{override_grammar}")

            except:
                logger.exception(f"An Exception occurred in '{function_name}' function while trying to override its grammar: {e}")
                print(f"Check the override grammar file. The override grammar must have no # comments or excess whitespace.")

        # Override both prompt and grammar.
        else:
            try:
                prompt = format_external_text_like_f_string(override_prompt, prompt_content)
                grammar = LlamaGrammar.from_string(fr"{override_grammar}")

            except:
                logger.exception(f"An Exception occurred in '{function_name}' function while trying to override its prompt and grammar: {e}")
                print(f"Note: the override prompt's text' must contain {list(prompt_content.keys())} somewhere in curly brackets.\nThe override grammar must have no # comments or excess whitespace.")

    except:
         logger.exception(f"An Exception occurred in 'override_prompt_and_grammar' function: {e}")

    return prompt, grammar
"""

def random_name():
    return random.choice(NAMES)


def sanity_check(LLM: dict) -> None:
    initialized_model = LLM['llm']
    retries = 0
    while retries <= 4:
        decision_prompt = f"""Hi there, """
        logger.info("DEBUG\n\n" + decision_prompt) if get_config("DEBUG_MODE") else None
        completion = initialized_model(
            decision_prompt,
            max_tokens=100,
            stop=["</s>", "# Input:"],
            echo=True,
            grammar="answer_accurate_grammar",
            temperature=0.2,
        )["choices"][0]["text"]
        logger.info(completion) if get_config("DEBUG_MODE") else None

        return

def strip_steps(instruction_text: str):
    """
    This function takes a string containing step-by-step instructions and removes the "Step N." prefix from each line.

    Parameters:
    instruction_text (str): A string with each step in the format "Step N. Instruction", separated by newlines

    Returns:
    str: A single string with the steps stripped, joined by newlines.
    """
    instructions = instruction_text.split("\n")
    stripped_instructions = []

    for line in instructions:
        # Check if line starts with 'Step' and followed by a number and period
        if line.strip().startswith("Step") and "." in line:
            # Find the index of the first period
            period_index = line.find(".")
            # Extract the text after the period (and optional space)
            text_after_period = line[period_index + 1 :].lstrip()
            stripped_instructions.append(text_after_period)
        else:
            stripped_instructions.append(line)

    return "\n".join(stripped_instructions)

def write_output_to_file(output, directory: str, uuid: str):
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path using the directory and UUID
    file_path = os.path.join(directory, f"{uuid}.txt")

    # Write the output to the file
    with open(file_path, "w", encoding='utf-8') as file:
        file.write(output)

    logger.info(f"Output written to {file_path}")

#########################################
######## NODE-SPECIFIC FUNCTIONS ########
#########################################


#########################################################
#### FUNCTIONS FROM ensure_multiple_answers_are_same ####
#########################################################

def has_sequential_chars(string1: str, string2: str, n: int):
    """
    Check if any n sequential characters from string1 appear in string2.

    Args:
    string1 (str): The first string to check.
    string2 (str): The second string in which to look for sequences.
    n (int): The length of the sequence to check.

    Returns:
    bool: True if any n sequential characters from string1 are found in string2, False otherwise.
    """

    # Check if n is larger than the length of string1.
    if n > len(string1):
        return False, ""

    # Iterate over string1 and check for each n-length substring in string2
    comparison_string = ""

    for i in range(len(string1) - n + 1):
        comparison_string = string1[i : i + n]

        if comparison_string in string2:
            return True, comparison_string

    return False, comparison_string

def extract_conversation(conversation: str) -> str:
    """
    Extracts conversation from a string and returns it as a list of tuples.

    Parameters:
    conversation (str): A string representing the conversation.

    Returns:
    list of tuples: Each tuple contains the character's name and their message.
    """
    lines = conversation.strip().split("\n")
    if len(lines) == 1: # If no newlines, there's 1 item
        lines = conversation.replace("## Conversation that answers the provided questions:",'').strip().split(r"\n")[1:]
    dialogues = []

    for line in lines:
        if ":" in line:
            # Splitting at the first occurrence of ':'
            parts = line.split(":", 1)
            charname = parts[0].strip()
            message = parts[1].strip() if len(parts) > 1 else ""
            dialogues.append((charname, message))

    return dialogues

def compare_answers_with_qatuples(dialogues: list, qatuples: list, n: int) -> bool:
    """
    Compares each answer in dialogues with the corresponding answer from qatuples.

    Parameters:
    dialogues (list): List of tuples containing the dialogues.
    qatuples (list): List of tuples containing questions and answers.
    n (int): Number of sequential characters to check.

    Returns:
    bool: True if all answers match the corresponding answers in qatuples, False otherwise.
    """
    for i in range(2, len(dialogues), 2):  # Answers are at even indices, starting from 2

        if int(i / 2) - 1 >= len(qatuples):  # at this point we've reached added stuff that doesn't have a corresponding qatuple
            break

        sequential, comp = has_sequential_chars(qatuples[int(i / 2) - 1][1], dialogues[i][1], n)
        logger.info(sequential)
        logger.info(n)

        if not sequential:
            logger.warning(f"Answer {int(i/2)}: {dialogues[i][1]} does not match the corresponding answer in qatuples: {qatuples[int(i/2) - 1][1]}, {comp}")
            return False

    return True

def check_for_repeated_dialogue_answers(dialogues: list, qatuples: list, n: int) -> bool:
    """
    Checks each line of dialogue to ensure that it does not repeat the corresponding answer from qatuples.

    Parameters:
    dialogues (list): List of tuples containing the dialogues.
    qatuples (list): List of tuples containing questions and answers.
    n (int): Number of sequential characters to check for repetition.

    Returns:
    bool: True if no dialogue line repeats its corresponding answer, False otherwise.
    """
    for i in range(2, len(dialogues), 2):  # Answers are at even indices, starting from 2

        if int(i / 2) - 1 >= len(qatuples):  # at this point we've reached added stuff that doesn't have a corresponding qatuple
            break

        dialogue_answer = dialogues[i][1]
        corresponding_qatuple_answer = qatuples[int(i / 2) - 1][1]

        # Check if the dialogue answer repeats the qatuple answer
        if dialogue_answer.count(corresponding_qatuple_answer) > 1:
            return False

    return True


def check_conversation_length(conv: str, qatuples: list) -> bool:
    """Checks the length of the conversation"""
    # Dialogues with answers should be at even indices that are not 0
    # qatuples are of the format (question, answer,source_text,name_of_text) -- only the first two are used here

    # Get the length of the dialogues
    conv_length = len(conv)

    target_length = len(qatuples) * 2 + 1

    if (conv_length < target_length):  # we can have more messages since the AI might add some stuff at the end to wrap up the scene
        return False
    else:
        return True

def check_conversation_for_text_from_examples(conv: str) -> bool:
    """Checks if certain strings from the few-shot examples appear in the conversation"""
    strings_to_check_for = [
        "her lipstick-colored lips", # lipstick-colored lips? Who writes this crap?
        "coquettishly tilts her head to the side,",
        "Awwww, you're no fun,",
        "Reminds me of my colleagues...",
        "" "I'll see you at that cafe.",
        "Ghh... you know,",
        "you're breaking a poor woman's heart,",
        "surprising innocence and warmth",
        'in mock-thought, "',
        " _I can't believe my ears. Did ",
    ]

    matches_found = 0
    for string in strings_to_check_for:
        if string in conv:
            matches_found += 1
            logger.info(f"Found {string} in the conversation!")

    if matches_found > 2:
        logger.warning(f"Found {matches_found} matches for strings from the few-shot examples. Validation failed!")
        return False

    return True

def check_each_question_contains_q_from_tuples(conv: str, qatuples: list, n: int) -> Union[bool, None]:
    """
    Ensures that each question contains at least n sequential characters from the corresponding question in qatuples.
    If the first question fails this check, return None for special handling.

    Parameters:
    conv (list): List of tuples containing the dialogues.
    qatuples (list): List of tuples containing questions and answers.
    n (int): Number of sequential characters to check.

    Returns:
    bool or None: True if all questions pass the check, False if any fail, None if the first question fails.
    """
    for i in range(1, len(conv), 2):  # Questions are at odd indices

        if i // 2 < len(qatuples):  # Ensure we only check questions that have corresponding qatuples
            question_from_conv = conv[i][1]
            question_from_tuples = qatuples[i // 2][0]
            # print(question_from_tuples, question_from_conv)
            sequential, _ = has_sequential_chars(question_from_tuples, question_from_conv, n)

            if not sequential:
                if i == 1:
                    return None  # Special handling for the first question
                else:
                    return False

    return True

def check_for_unintended_repeated_quotes(dialogues: list, qatuples: list, n_characters_shared: int) -> bool:
    """
    Checks if answers in the conversation inadvertently use a long quote from another QA pair.

    Args:
    dialogues (list): List of tuples containing the dialogues.
    qatuples (list): List of tuples containing questions and answers.
    n_characters_shared (int): Number of sequential characters to check for repetition.

    Returns:
    bool: True if no unintended repeated quotes are found, False otherwise.
    """

    # Extract only the answers from the QA tuples for comparison
    qa_answers = [qa[1] for qa in qatuples]

    for i in range(2, len(dialogues), 2): # Answers are at even indices, starting from 2
        # Skip if there's no corresponding QA tuple
        if int(i / 2) - 1 >= len(qatuples):
            break

        dialogue_answer = dialogues[i][1]
        corresponding_qa_answer = qatuples[int(i / 2) - 1][1]

        # Check for each answer in the QA tuples
        for idx, qa_answer in enumerate(qa_answers):
            # Skip the comparison for the current QA pair itself
            if qa_answer == corresponding_qa_answer:
                continue

            # Check if the dialogue answer contains a long quote from another QA answer
            sequential, comp_string = has_sequential_chars(
                qa_answer, dialogue_answer, n_characters_shared
            )
            if sequential:
                if comp_string in corresponding_qa_answer:
                    continue  # This is a quote from the corresponding answer, so it's fine
                else:
                    # Found an unintended repeated quote
                    return False
    return True

def call_all_processors(multiturn_conversation: str, qatuples: list) -> bool:
    convs_split = extract_conversation(multiturn_conversation)

    # Check if answers in dialogues match corresponding answers in qatuples
    if not compare_answers_with_qatuples(convs_split, qatuples, 15):
        logger.warning("Answers in dialogues do not match corresponding answers in qatuples.")
        return False

    # Check if any dialogue line repeats its corresponding answer
    if not check_for_repeated_dialogue_answers(convs_split, qatuples, 15):
        logger.warning("Dialogue line repeats its corresponding answer.")
        return False

    # Check the conversation length
    if not check_conversation_length(convs_split, qatuples):
        logger.warning("Conversation is too short! Validation failed!")
        return False

    # Check for text from examples (assuming this is implemented elsewhere)
    if not check_conversation_for_text_from_examples(multiturn_conversation):
        logger.warning("Conversation does not contain text from examples. Validation failed!")
        return False

    # Check for unintended repeated quotes
    if not check_for_unintended_repeated_quotes(convs_split, qatuples, 100):
        logger.warning("Conversation contains unintended repeated quotes. Validation failed!")
        return False

    # Check each question contains a part of the question from tuples
    result = check_each_question_contains_q_from_tuples(convs_split, qatuples, 15)
    if result is None:
        logger.warning("First question does not contain a part of the question from tuples. Validation failed!")
        return None
    elif not result:
        logger.warning("Each question does not contain a part of the question from tuples. Validation failed!")
        return False

    # If all checks pass
    return True


######################
#### NODE CLASSES ####
######################


#TODO Write function documentation.
#TODO Add in LLM preset override functionality.
#TODO Investigate function and class set-up. Something seems wrong here...
class ReviseQATuples:
    """
    This function revises a QA tuple. It's really a fuck-ton of functions in a trench-coat.

    :param vetted_qa_tuples: Output from the JudgeParagraphs function, originally called "filtered_worthy_for_questions".
    :param LLM: An initialized model and its presets.
    :param qa_tuple_directory_name: The name of directory where the AQ tuples will go.
    :param total_retries: The total number of retries the function should run.
    :return vetted_qa_tuples: List of sentence chunks with source text information
    """

    @staticmethod
    def extract_question_answer(response: str):
        # Define the regex pattern to match the question and answer
        pattern = r"### Question Rewording \(using text details as reference\):\nQuestion: (.+?)\nAnswer: (.+)"

        # Search for the pattern in the response
        match = re.search(pattern, response)

        # Extract and return the question and answer if a match is found
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            return question, answer
        else:
            logger.warning("Returned none, failed to match")
            return None, None

    @staticmethod
    def check_qatuple_context(qatuple: Tuple, LLM: dict):
        retries = 0
        initialized_model = LLM['llm']
        prompt_content = {
            "qatuple": qatuple,
        }

        # Load the prompt and grammar.
        try:
            decision_prompt, check_qatuple_context_grammar = load_external_prompt_and_grammar("check_qatuple_context","check_qatuple_context_grammar", prompt_content)
        except Exception as e:
            logger.exception(f"An Exception occured in 'check_qatuple_context' function while loading its prompt and grammar: {e}")

        # Load the override LLM dictionary, if it exists. If not, use the function's defaults.
        try: 
            overrides = LLM['override_check_qatuple_context_presets']

           # Override the default function presets if it's requested.
            if overrides.get('override_llm_presets') is True:
                logger.info("Overriding default LLM presets for 'check_qatuple_context' function.")
                initialized_model = overrides['llm']
                LLM['override_llm_presets'] = True

            # Override the prompt if it's requested.
            if overrides.get('prompt') is not None:
                decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
                logger.info("Overriding the prompt for 'check_qatuple_context' function.")

            # Override the grammar if it's requested.
            if overrides.get('grammar') is not None: 
               check_qatuple_context_grammar = overrides['grammar']
               logger.info("Overriding the grammar for 'check_qatuple_context' function.")

        except KeyError:
            logger.info("Overrides for 'check_qatuple_context' function not present. Using default presets.")

        while retries <= 4:

            try: # Check the QA tuple's context using the LLM.
                start_time = time.time()
                logger.info("Generating 'check_qatuple_context' completion for qatuple...")

                if LLM['override_llm_presets'] == "True":
                    completion = initialized_model(
                        decision_prompt,
                        check_qatuple_context_grammar
                    )["choices"][0]["text"]
                else:
                    completion = initialized_model(
                        decision_prompt,
                        max_tokens=10000,
                        stop=["</s>", "# Input:"],
                        echo=True,
                        grammar=check_qatuple_context_grammar,
                        temperature=0.2,
                    )["choices"][0]["text"]

                end_time = time.time()
                if get_config("DEBUG_MODE"):
                    logger.info(f"Completion for 'generate_questions_plan' function ***\n{completion}\n*** Completion for 'generate_questions_plan' function")

                logger.info(f"Completion took {(end_time - start_time) / 60} minutes to generate.")
                logger.info(f"Completion for 'generate_questions_plan' function for retry {retries} generated. Extracting response pattern...")

                # Extract the response pattern from the LLM's completion.
                response_pattern = re.compile(
                    r"Reasoning and thought process \(be thorough\):(.+)", 
                    re.DOTALL | re.IGNORECASE,
                )
                response = response_pattern.search(completion).group(1).strip()

                # Extract the decision pattern from the LLM's response pattern.
                decision_pattern = re.compile(r"Final judgment:(.+)", re.IGNORECASE)
                if get_config("DEBUG_MODE"):
                    logger.info(f"Response for 'check_qatuple_context' function ***\n{response}\n*** Response for 'check_qatuple_context' function ***")

                # Extract the determination from the LLM's decision pattern.
                determination = decision_pattern.search(response).group(1).strip()
                logger.info(f"\n\nDETERMINATION:\n------\n{determination}\n---------\n")

                # Revise the QA tuple if the LLM says it needs rewording.
                if "Reword" in determination or "reword" in determination.lower():
                    logger.info("Rewording...")
                    q, a = ReviseQATuples.extract_question_answer(response)
                    logger.info((q, a, qatuple[2], qatuple[3]))
                    return (q, a, qatuple[2], qatuple[3]), completion

                # Skip the QA tuple if the LLM says it passes.
                elif "Pass" in determination or "pass" in determination.lower():
                    logger.info("Leaving be...")
                    return (True, response), completion

                # Set the QA tuple to None if the LLM says it fails.
                elif "Fail" in determination or "fail" in determination.lower():
                    logger.info("Setting to None...")
                    return (False, response), completion

                # Retry everything if the LLM didn't do its job.
                else:
                    logger.info(f"Did not contain relevant or irrelevant in retry {retries}! Retrying")
                    retries += 1

            except Exception as e:
                logger.exception(f"An Exception occurred in check_qatuple_context function in class ReviseQATuples: {e}")

                if retries <= 4:
                    retries += 1
                else:
                    return (None, None), None

        return (None, None), None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "vetted_qa_tuples": ("TUPLE", {"forceInput": True}),
                "LLM": ("LLM",),
                "qatuple_directory_name": ("STRING", {"default": 'qatuples_raw'}),
            },
            "optional":{
                "override_check_qatuple_context_presets": ("LLM",),
            },
        }
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("vetted_qa_tuples",)
    FUNCTION = "revise_qa_tuples"

    CATEGORY = "augmentoolkit_functions"

    def revise_qa_tuples(self, vetted_qa_tuples, LLM, qatuple_directory_name,override_check_qatuple_context_presets=None):
        # Check for and fix the common mistake: mentioning "the text".
        # TODO refactor to be continuable, should take like 30 mins at most

        if override_check_qatuple_context_presets:
            LLM['override_check_qatuple_context_presets'] = override_check_qatuple_context_presets

        qatuples_revised_directory = f"./{qatuple_directory_name}"

        # Assuming vetted_qa_tuples is a list that might or might not exist
        try:
            _ = vetted_qa_tuples
        except NameError:
            vetted_qa_tuples = []


        # Load all files at the start if vetted_qa_tuples is empty
        if not vetted_qa_tuples:
            # Check if the directory exists
            if os.path.exists(qatuples_revised_directory):
                # List all files in directory
                for file_name in os.listdir(qatuples_revised_directory):
                    file_path = os.path.join(qatuples_revised_directory, file_name)

                    try: #Open the files
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            logger.info(f"Loading file: {file_path}")

                            if content == "failed": # Skip the files with content that failed.
                                vetted_qa_tuples.append(None)
                            else:
                                try: # Put the file data into the vetted_qa_tuples list.
                                    data = json.loads(content)
                                    vetted_qa_tuples.append(
                                        (data[0], data[1], data[2], data[3])
                                    )
                                except json.JSONDecodeError:
                                    logger.error(f"JSON decode error with the contents: {content}")
                                    vetted_qa_tuples.append(None)

                    except Exception as e:
                        logger.exception(f"An exception occured reading {file_path} in revise_qa_tuples function in class ReviseQATuples: {e}")

        else:
            if LLM['type'] == 'llamacpp': # The original Llama-cpp loop.
                old_tuples = vetted_qa_tuples.copy()

                for idx, tup in enumerate(vetted_qa_tuples):
                    file_path = os.path.join(qatuples_revised_directory, f"revised_{idx}.json")

                    if os.path.exists(file_path):
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()  # Read the file once and store its content
                            logger.info(f"Loading file: {file_path}")

                            if content == "failed":
                                logger.warning("Loaded failed file")
                                vetted_qa_tuples[idx] = None
                                continue
                            logger.info(f"Loaded file:\n{content}")

                            # Reset the file pointer to the beginning if you need to read again or convert string back to JSON
                            try:
                                data = json.loads(content)  # Convert the string back to JSON
                                vetted_qa_tuples[idx] = (data[0], data[1], data[2], data[3])
                                continue
                            except json.JSONDecodeError:
                                # Raise a JSON decode error if it can't dump the data to json.
                                logger.error(f"JSON decode error with the contents in revise_qa_tuples function in class ReviseQATuples: {content}")
                                raise json.JSONDecodeError

                    try:
                        revision_id = make_id()
                        revision, revision_output = self.check_qatuple_context(tup, LLM)
                        write_output_to_file(
                            revision_output, "./question_context_revision_generations", revision_id
                        )  # incidentally, identifying the problem and fixing it in the same step (without another planning step) works a lot better than identifying it and then trying to fix it in the next step.

                        if isinstance(revision[0], str): vetted_qa_tuples[idx] = revision # if the thing was reworded
                        elif not revision[0]: vetted_qa_tuples[idx] = None  # prepare item for deletion later; right now we just store it as None because indexes
                        # else, if it passed, we just leave it be.

                        # Write in-progress
                        if not os.path.exists(qatuples_revised_directory):
                            os.makedirs(qatuples_revised_directory)

                        if vetted_qa_tuples[idx]:
                            with open(file_path, "w") as file:
                                json.dump(vetted_qa_tuples[idx], file, indent=4)
                        else:
                            with open(file_path, "w") as file:
                                file.write("failed")
                                logger.warning("Failed to open vetted_qa_tuples file.")

                    except Exception as e:
                        logger.exception(f"An Exception occurred in revise_qa_tuples in class ReviseQATuples: {e}")
            # TODO Figure out which augmentoolkit code to use here.
            elif LLM['type'] == 'aphrodite' and APHRODITE_NOT_INSTALLED is False: # Aphrodite route.
                #old_tuples = vetted_qa_tuples.copy()

                # Set up tasks list before running asyncio
                tasks = [aug_async.generate_qatuples_from_para(
                    idx,
                    para,
                    engine_wrapper=LLM,
                    vetted_qa_tuples=vetted_qa_tuples,
                    qa_tuples_dir=qatuples_revised_directory,
                    double_check_counter=get_config("DOUBLE_CHECK_COUNTER"),
                    use_filenames=get_config("USE_FILENAMES")) for idx, para in enumerate(vetted_qa_tuples)
                 ]

                limited_tasks_qgen = [run_task_with_limit(task) for task in tasks]

                # Put the rate-limiter on beforing running asyncio
                limited_tasks_qcorrection = [run_task_with_limit(task) for task in tasks]
                asyncio.run(run_tasks(limited_tasks_qcorrection))

            elif LLM['type'] == 'api': #API route.
                # Set up tasks list before running asyncio
                tasks = [aug_async.generate_qatuples_from_para(
                    idx,
                    para,
                    engine_wrapper=LLM,
                    vetted_qa_tuples=vetted_qa_tuples,
                    qa_tuples_dir=qatuples_revised_directory,
                    double_check_counter=get_config("DOUBLE_CHECK_COUNTER"),
                    use_filenames=get_config("USE_FILENAMES")) for idx, para in enumerate(vetted_qa_tuples)
                 ]

                limited_tasks_qgen = [run_task_with_limit(task) for task in tasks]

                # Put the rate-limiter on beforing running asyncio
                limited_tasks_qcorrection = [run_task_with_limit(task) for task in tasks]
                asyncio.run(run_tasks(limited_tasks_qcorrection))
            else:
                pass

        # Print stats related to revised qatuples, and filter out nones (questions that were unanswerable due to lack of context).

        logger.info("-------------- QUESTIONS REVISED ------------- STATS SO FAR:")
        nones = list(filter(lambda x: x is None, vetted_qa_tuples))
        logger.info(f"\nNones: {len(nones)}\nNon-nones: {len(vetted_qa_tuples) - len(nones)}\nTotal: {len(vetted_qa_tuples)}")

        # filter out all None values
        vetted_qa_tuples = [qa for qa in vetted_qa_tuples if qa is not None]
        logger.info("---------------- ONTO EXAMPLES GENERATION-------------------")

        return(vetted_qa_tuples,)



############################################
#### GenerateQATuplesAdvanced Functions ####
############################################

# Wrapper for the async 'tqdmasyncio' function object, since this function needs to be called in non-async environments.
async def run_tasks(tasks):
    for future in tqdmasyncio.tqdm.as_completed(tasks):
        await future

def check_answer(qatuple: Tuple, LLM: dict, permissive_mode=True):  # The fuck is permissive_mode???
    # Initialize variables
    initialized_model = LLM['llm']
    retries = 0
    prompt_content = {
        "qatuple": qatuple,
    }

    # Load the prompt and the grammar.
    try:
        decision_prompt, answer_accurate_grammar = load_external_prompt_and_grammar("check_answer", "answer_accurate_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in 'check_answer' function while trying to import its prompt and grammar: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_check_answer_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_llm_presets') is True:
            logger.info("Overriding default LLM presets for 'check_answer' function.")
            initialized_model = overrides['llm']
            LLM['override_llm_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            logger.info("Overriding the prompt for 'check_answer' function.")

        # Override the grammar if it's requested.
        if overrides.get('grammar'): 
            answer_accurate_grammar = overrides['grammar']
            logger.info("Overriding the grammar for 'check_answer' function.")

    except KeyError:
        logger.info("Overrides for 'check_answer' function not present. Using default presets.")

    while retries <= 4:
        # Load the initialized LLM and check the accuracy of the answer in the QA tuple.
        try:
            start_time = time.time()
            logger.info(f"Generating 'check_answer' completion... \nCurrent Retry Count: {retries}")

            # Check if the LLM's settings have been overridden, then route appropriately.
            if LLM['override_llm_presets'] is True:
                completion = initialized_model(
                    decision_prompt, 
                    grammar=answer_accurate_grammar
                )["choices"][0]["text"]
            else:
                completion = initialized_model(
                    decision_prompt,
                    max_tokens=6000,
                    stop=["</s>", "# Input:"],
                    echo=True,
                    grammar=answer_accurate_grammar,
                    temperature=0.2,
                )["choices"][0]["text"]

            end_time = time.time()
            logger.info(f"Completion took {(end_time - start_time) / 60} minutes to complete.")
            logger.info(f"Completion for 'check_answer' function on retry {retries} generated. Extracting response pattern...")

            completion_pattern = re.compile(
                r"Reasoning and thought process \(the text is your single source of truth\):([\s\S]*)", 
                re.DOTALL,
            )

            # Extract the response pattern from the LLM's completion.
            response = completion_pattern.search(completion).group(1).strip()
            if get_config("DEBUG_MODE"):
                logger.info(f"*** Response for 'check_answer' function ***\n{response}\n *** Completion for 'check_answer' function ***")

            # Extract the LLM's determination from the response pattern.
            # Permissive mode is TODO: Figure out what the fuck permissive mode is.
            if permissive_mode:
                determination_pattern = re.compile(
                    r"Overall Accuracy Determination:([\s\S]*)", 
                    re.DOTALL,
                )
                determination = determination_pattern.search(response).group(1).strip()
                logger.info(f"\n\nDETERMINATION: PERMISSIVE MODE:\n------\n{determination}\n---------\n")
            else:
                determination = response
                logger.info(f"\n\nDETERMINATION:\n------\n{determination}\n---------\n")

            # Determine whether the answer is accurate, inaccurate, or needs to go back into the while loop for another check.
            if (
                "inaccurate" in determination.lower()
                or "Inaccurate" in determination.lower()
                or "mostly" in determination.lower()
                or "partial" in determination.lower()
            ):  # The "mostly" is there to catch "mostly accurate" which the model says occasionally, and which actually means inaccurate. Can't this be fixed in the grammar?
                return (False, response), completion

            elif ("accurate" in determination or "Accurate" in determination):  # very deliberate placement of accurate here, because the model can sometimes say irrelevant at the very end, even after saying accurate in its judgment. Gotta love temperature, am I right?
                return (True, response), completion

            elif ("irrelevant" in determination or "Irrelevant" in determination):  # optional support for checking relevance here, too.
                return (None, response, ), completion  # signal that question is irrelevant

            else:
                logger.error("Broke!") # What the fuck does this mean???

        except Exception as e:
            retries += 1
            logger.exception(f"An Exception occurred in check_answer function in class GenerateQATuplesAdvanced: {e}")

def check_answer_relevancy_with_text(qatuple: Tuple, LLM: dict): #TODO: Document this function.
    retries = 0
    initialized_model = LLM['llm']
    prompt_content = {
        "text": qatuple[2],
        "question": qatuple[0],
        "answer": qatuple[1],
    }

    # Load the prompt and the grammar.
    try:
        decision_prompt, answer_relevant_grammar = load_external_prompt_and_grammar("check_answer_relevancy_with_text", "answer_relevant_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in 'check_answer_relevancy_with_text' function while trying to import its prompt and grammar: {e}")

    try: # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
        overrides = LLM['check_answer_relevancy_with_text']

        if overrides.get('override_llm_presets') is True: # Override the default function presets if it's requested.
            logger.info("Overriding default LLM presets for 'check_answer_relevancy_with_text' function.")
            initialized_model = overrides['llm']
            LLM['override_llm_presets'] = True

        if overrides.get('prompt'): # Override the prompt if it's requested.
            decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            logger.info("Overriding the prompt for 'check_answer_relevancy_with_text' function.")

        if overrides.get('grammar'): # Override the grammar if it's requested.
            answer_relevant_grammar = overrides['grammar']
            logger.info("Overriding the grammar for 'check_answer_relevancy_with_text' function.")

    except KeyError:
        logger.info("Overrides for 'check_answer_relevancy_with_text' function not present. Using default presets.")

    while retries <= 4:
        # Load the initialized LLM and check the QA tuple's answer's relevancy to the question by comparing it to the original text.
        try:
            start_time = time.time()
            logger.info(f"Generating 'check_answer_relevancy_with_text' completion... \nCurrent Retry Count: {retries}")

            if LLM['override_llm_presets'] is True:
                completion = initialized_model(
                    decision_prompt,
                    grammar=answer_relevant_grammar
                )["choices"][0]["text"]
            else:
                completion = initialized_model(
                    decision_prompt,
                    max_tokens=5500,
                    stop=["</s>", "# Input:"],
                    grammar=answer_relevant_grammar,
                    echo=True,
                    temperature=0.2,
                )["choices"][0]["text"]

            end_time = time.time()
            logger.info(f"Completion for 'check_answer_relevancy_with_text' function generated in {(end_time - start_time) / 60} minutes.")
            logger.info(f"Completion for 'check_question' function on retry {retries} generated. Extracting response pattern...")

            completion_pattern = re.compile(
                r"Reasoning and thought process \(be careful about extra details, even vague ones\):([\s\S]*)", 
                re.DOTALL | re.IGNORECASE,
            )
            judgement_pattern = re.compile(
                r"Explanation of Judgment:([\s\S]*)", 
                re.DOTALL | re.IGNORECASE
            )

            # Extract the response pattern from the LLM's completion.
            response = completion_pattern.search(completion).group(1).strip()
            if get_config("DEBUG_MODE"):
                logger.info(f"Response for 'check_answer_relevancy_with_text' function on retry {retries} ***\n{response}\n *** Completion for 'check_answer_relevancy_with_text' function on retry {retries}")

            # Extract the LLM's determination from the response pattern.
            determination = judgement_pattern.search(response).group(1).strip()
            logger.info(f"\n\nDETERMINATION:\n------\n{determination}\n---------\n")

            # Determine whether the answer is relevant to the text, irrelevant to the text, or needs to go back into the while loop for another check.
            if (
                "irrelevant" in determination.lower()
                or "mostly" in determination.lower()
                or "partial" in determination.lower()
                or "introduces information not present in the text"
                in determination.lower()
            ):  # Hack to get around faulty 13b outputs
                return (False, response), completion
            elif "relevant" in determination or "Relevant" in determination:
                return (True, response), completion
            else:
                retries += 1

        except:
            retries += 1
            logger.exception(f"Exception in text completion from check_answer_relevancy_with_text function. Investigate! Here's the completion:\n{completion}")

def check_question(qatuple: Tuple, LLM: dict): #TODO: Document this function.
    retries = 0
    initialized_model = LLM['llm']
    prompt_content = {
        "qatuple": qatuple,
    }

    # Load the prompt and the grammar.
    try:
        decision_prompt, question_relevant_grammar = load_external_prompt_and_grammar("check_question", "question_relevant_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in 'check_question' function while trying to import its prompt and grammar: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_check_question_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_llm_presets') is True:
            logger.info("Overriding default LLM presets for 'check_question' function.")
            initialized_model = overrides['llm']
            LLM['override_llm_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            logger.info("Overriding the prompt for 'check_question' function.")

        # Override the grammar if it's requested.
        if overrides.get('grammar'): 
            question_relevant_grammar = overrides['grammar']
            logger.info("Overriding the grammar for 'check_question' function.")

    except KeyError:
        logger.info("Overrides for 'check_question' function not present. Using default presets.")

    while retries <= 4:

        # Load the initialized LLM and check the question in the QA tuple.
        try:
            start_time = time.time()
            logger.info(f"Generating 'check_question' completion... \nCurrent Retry Count: {retries}")

            if LLM['override_llm_presets'] is True:
                completion = initialized_model(
                    decision_prompt,
                    grammar=question_relevant_grammar
                )["choices"][0]["text"]
            else:
                completion = initialized_model(
                    decision_prompt,
                    max_tokens=4000,
                    stop=["</s>", "# Input:"],
                    echo=True,
                    grammar=question_relevant_grammar,
                    temperature=0.2,
                )["choices"][0]["text"]

            end_time = time.time()
            logger.info(f"Completion took {(end_time - start_time) / 60} minutes to generate.")
            logger.info(f"Completion for 'check_question' function on retry {retries} generated. Extracting response pattern...")

            response_pattern = re.compile(
                r"Reasoning and thought process \(be careful around \"how\" and \"why\" questions\):([\s\S]*)",
                re.DOTALL | re.IGNORECASE,
            )

            decision_pattern = re.compile(
                r"Final Judgment:([\s\S]*)", 
                re.DOTALL | re.IGNORECASE,
            )

            # Extract the response pattern from the LLM's completion.
            response = response_pattern.search(completion).group(1).strip()
            if get_config("DEBUG_MODE"):
                logger.info(f"\n*** Response for 'check_question' function ***\n{response}\n*** Response for 'check_question' function ***\n")

            # Extract the LLM's determination from the response pattern.
            determination = decision_pattern.search(response).group(1).strip()
            logger.info(f"\n\nDETERMINATION:\n------\n{determination}\n---------\n")

            # Determine whether the question is relevant, irrelevant, or needs to go back into the while loop for another check.
            if (
                "irrelevant" in determination
                or "Irrelevant" in determination.lower()
                or "mostly" in determination.lower()
                or "partial" in determination.lower()
                or "introduces information not present in the text"
                in determination.lower()
            ):
                return (False, response), completion
            elif "relevant" in determination or "Relevant" in determination:
                return (True, response), completion
            else:
                logger.info(f"Did not contain relevant or irrelevant in retry {retries}! Retrying")
                retries += 1

        except Exception as e:
            logger.exception(f"An Exception occurred in 'check_question' function in class GenerateQATuples: {e}")
            # Screw stopping! If something breaks here, we keep going damnit!
            if retries <= 4:
                retries += 1
            else:
                return (None, None), completion

    return (None, None), None

def generate_new_question(qatuple: Tuple, LLM: dict):
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    """
    # Determine which paragraphs are worthy of making questions from
    made_questions = False
    initialized_model = LLM['llm']
    retries = 0
    questions = []
    prompt_content = {
        "qatuple": qatuple,
    }

    # Load the prompt and the grammar.
    try:
        question_prompt, question_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "question_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in 'generate_new_question' function while trying to import its prompt and grammar: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_generate_new_question_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_llm_presets') is True:
            logger.info("Overriding default LLM presets for 'generate_new_question' function.")
            initialized_model = overrides['llm']
            LLM['override_llm_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            question_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            logger.info("Overriding the prompt for 'generate_new_question' function.")

        # Override the grammar if it's requested.
        if overrides.get('grammar'): 
            question_grammar = overrides['grammar']
            logger.info("Overriding the grammar for 'generate_new_question' function.")

    except KeyError:
        logger.info("Overrides for 'generate_new_question' function not present. Using default presets.")

    while not made_questions and (retries <= 5):  # TODO - UPDATE and TEST the few-shot prompt with the latest from generate_questions

        logger.info(f"--QA TUPLE DURING NEW Q GEN--\n{qatuple}")
        start_time = time.time()
        logger.info(f"Generating 'generate_new_question' function completion... \nCurrent Retry Count: {retries}")

        if LLM['override_llm_presets'] is True:
            completion = initialized_model(
                question_prompt, 
                grammar=question_grammar
            )["choices"][0]["text"]
        else:
            completion = initialized_model(
                question_prompt,
                max_tokens=8000,
                stop=["</s>", "# Input"],
                echo=True,
                grammar=question_grammar,
                temperature=0.2,
            )["choices"][0]["text"]

        end_time = time.time()
        logger.info(f"Completion took {(end_time - start_time) / 60} minutes to generate.")
        logger.info(f"Completion for 'generate_new_question' function on retry {retries} generated. Extracting response pattern...")

        # Extract questions
        response_pattern = re.compile(
            r"Question \(based on text\):([\s\S]*)", 
            re.IGNORECASE | re.DOTALL
        )
        generation = response_pattern.search(completion).group(1)
        if get_config("DEBUG_MODE"):
            logger.info(f"\nResponse for 'generate_new_question' function ***\n{generation}\n*** Response for 'generate_new_question' function ***")

        print("-------------------")

        pattern = re.compile(
            r"(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)", 
            re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        matches = pattern.findall(generation)
        logger.info(f"\nMatches for 'generate_new_question' function ***\n{matches}\n*** Matches for 'generate_new_question' function ***")

        if len(matches) > 0:
            logger.info("Made Qs, yay!")
            made_questions = True
        else:
            logger.info("No matches found. Retry!")
            retries += 1

    for match in matches:
        return (
            match[0].replace(") ", "", 1).strip(),
            match[1].replace(") ", "", 1).strip(),
            qatuple[2].replace(") ", "", 1),
            qatuple[3],
        ), completion
    logger.warning(f"Should not have reached here\n{matches}\n{questions}")

    return questions, completion

def generate_qa_tuples(LLM: dict, 
                       qa_tuple_directory_name: str, 
                       question_plan_directory_name: str, 
                       question_plan_generations_directory_name: str,
                       total_retries: int, 
                       filtered_worthy_for_questions=None,):
    """
    This function generates QA tuples from input paragraphs.

    :param filtered_worthy_for_questions: Filtered Output from the JudgeParagraphs node. Default format: tuples of a paragraph chunk and source meta-data.
    :param LLM: An initialized model and its presets.
    :param qa_tuple_directory_name: The name of directory where the QA tuples will go.
    :param total_retries: The total number of retries the QA tuples should be cross-checked before being saved to output.
    :return vetted_qa_tuples: A list of QA tuples, vetted for relevance and accuracy.
    """
    # Set directory for QA tuples, and make it if it doesn't exist.
    qa_tuples_dir = f"./{qa_tuple_directory_name}"
    if not os.path.exists(qa_tuples_dir):
        os.makedirs(qa_tuples_dir)

    # Initialize vetted_qa_tuples list
    vetted_qa_tuples = []  # tuple list of QA tuples that have been judged good

    # Attempt to initialize filtered_worthy_for_questions list. 
    # If NameError occurs, create an empty array for the filtered_worthy_for_questions variable
    try:
        _ = filtered_worthy_for_questions
    except NameError:
        filtered_worthy_for_questions = []

    if filtered_worthy_for_questions is None:
        # Load all files in the qa_tuples_dir if filtered_worthy_for_questions is not initialized
        existing_files = glob.glob(os.path.join(qa_tuples_dir, "*.json"))

        # Load QA tuples from an external json 
        for file_path in existing_files:
            with open(file_path, "r") as file:
                qa_tuple = tuple(json.load(file))
            vetted_qa_tuples.append(qa_tuple)

    else:
        if LLM['type'] == 'llamacpp':
            # For every paragraph and their index number in the judged paragraphs...
            for idx, para in enumerate(tqdm(filtered_worthy_for_questions)):

                try:
                    existing_files = glob.glob(os.path.join(qa_tuples_dir, f"para_{idx}_*.json"))  # check if questions already exist

                    # Load in the QA tuple json if it exists already.
                    if len(existing_files) > 0:  # If files exist, skip this paragraph entirely
                        logger.info(f"Skipping para_{idx} as files already exist; loading said files...")

                        # For every file that already exists, open them and put them in the vetted_qa_tuples list.
                        for file_path in existing_files:
                            with open(file_path, "r") as file:
                                qa_tuple = tuple(json.load(file))
                            vetted_qa_tuples.append(qa_tuple)
                        continue

                    # Give each question group a unique id.
                    question_group_id = make_id()

                    # Create a question plan from the input paragraph, then write it to a file.
                    logger.info(f"\n\n\nOUTER LOOP CALL GENERATE QPLAN \npara: {para}, \n\n idx: {idx}")
                    plan, questions_plan_output = generate_questions_plan(para, LLM)
                    write_output_to_file(questions_plan_output, f"./{question_plan_directory_name}", question_group_id)

                    # Create QAs from the input paragraph and question plan, then write it to a file.
                    logger.info(f"\n\n\nOUTER LOOP CALL GENERATE Q: \npara: {para}, \n\n idx: {idx} \n\n plan: {plan}"        )
                    question_answer_tuples, question_generation_output = generate_questions(para, plan, LLM)
                    write_output_to_file(question_generation_output, f"./{question_plan_generations_directory_name}", question_group_id)

                    # Begin vetting the QA.
                    for qnum, question_answer_tuple in enumerate(question_answer_tuples):
                        logger.info(f"\n\n=======!!=BEGIN VETTING QA TUPLE {idx}_{qnum}=!!=======\n\n")
                        good_qa_tuple = vet_question_loop(
                            question_answer_tuple, 
                            LLM, 
                            total_retries, 
                            run_id=question_group_id,
                        )

                        # Write resulting question json file if the tuple is not None
                        if good_qa_tuple[0] is not None:
                            file_name = f"para_{idx}_q_{qnum}.json"
                            file_path = os.path.join(qa_tuples_dir, file_name)
                            with open(file_path, "w") as file:
                                json.dump(good_qa_tuple, file, indent=4)
                            logger.info(f"\n--------------\n*** Done! *** \nQA Tuple written to '{file_name}' in directory '{qa_tuples_dir}'\n--------------\n")

                        vetted_qa_tuples.append(good_qa_tuple)  # We must filter out all None values at the end; but appending Nones lets us know where things went wrong, and how often.

                except Exception as e:
                    logger.exception(f"An Exception occurred in generate_qa_tuples function: {e}")

        elif LLM['type'] == 'aphrodite':
            # Create a list of tasks.
            tasks = [aug_async.generate_qatuples_from_para(
                idx,
                para,
                engine_wrapper=LLM['llm'],
                vetted_qa_tuples=vetted_qa_tuples,
                qa_tuples_dir=qa_tuples_dir,
                double_check_counter=get_config("DOUBLE_CHECK_COUNTER")) for idx,para in enumerate(filtered_worthy_for_questions)
            ] # Schedule all the tasks. See the documentation in augmentoolkit_async_functions.py for more information.
            asyncio.run(run_tasks(tasks))

        elif LLM['type'] == 'api':
            pass
        else:
            pass

    # Create basic summary stats for the outputs so far, including None values.
    logger.info("-------------- QUESTIONS CREATED ------------- STATS SO FAR (may be wrong if run was continued from interruption):")
    nones = list(filter(lambda x: x[0] is None, vetted_qa_tuples))
    logger.info(f"\nNones: {len(nones)} \nNon-nones: {len(vetted_qa_tuples) - len(nones)} \nTotal: {len(vetted_qa_tuples)}")
    time.sleep(3)

    # filter out all None values
    vetted_qa_tuples = [qa for qa in vetted_qa_tuples if qa[0] is not None]

    return vetted_qa_tuples

def generate_questions(para_tuple: tuple, plan: str, LLM: dict): #TODO: Document this function.
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    :param para_tuple: A tuple consisting of a paragraph of source text and source meta-data.
    :param plan: A question plan created by the generate_questions_plan function.
    :param LLM: An initialized LLM (external).
    :return questions:
    :return completion:
    """
    # Determine which paragraphs are worthy of making questions from
    made_questions = False
    initialized_model = LLM['llm']
    retries = 0
    questions = []
    prompt_content = {
        "para_tuple": para_tuple,
        "strip_steps_plan": strip_steps(plan),
    }

    # Load the prompt and the grammar.
    try:
        question_prompt, questions_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "questions_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_generate_questions_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_llm_presets') is True:
            logger.info("Overriding default LLM presets for 'generate_questions' function.")
            initialized_model = overrides['llm']
            LLM['override_llm_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            question_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            logger.info("Overriding the prompt for 'generate_questions' function.")

        # Override the grammar if it's requested.
        if overrides.get('grammar'): 
            questions_grammar = overrides['grammar']
            logger.info("Overriding the grammar for 'generate_questions' function.")

    except KeyError:
        logger.info("Overrides for 'generate_questions' function not present. Using default presets.")


    while not made_questions and (retries <= 5):
        #print("DEBUG\n\n" + decision_prompt)

        start_time = time.time()
        logger.info(f"Generating 'generate_new_question' completion... \nCurrent Retry Count: {retries}")

        if LLM['override_llm_presets'] is True:
            completion = initialized_model(
                question_prompt, 
                grammar=questions_grammar
            )["choices"][0]["text"]
        else:
            completion = initialized_model(
                question_prompt,
                max_tokens=12000,
                stop=["</s>", "# Input:"],
                echo=True,
                grammar=questions_grammar,
                temperature=0.8,
                top_k=0,
                top_p=1,
                min_p=0.5,
            )["choices"][0]["text"]

        end_time = time.time()
        logger.info(f"Completion took {(end_time - start_time) / 60} minutes to generate.")
        logger.info(f"Completion for 'generate_questions' function on retry {retries} generated. Extracting response pattern...")

        # Extract questions from completion
        response_pattern = re.compile(r"Questions \(make 4\):([\s\S]*)", re.IGNORECASE | re.DOTALL)
        generation = response_pattern.search(completion).group(1)
        logger.info(f"Response for 'generate_questions' function ***\n{generation}\n*** Response for 'generate_questions' function ***")

        pattern = re.compile(
            r"(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)", 
            re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )

        matches = pattern.findall(generation)

        if len(matches) > 0:
            made_questions = True
        else:
            retries += 1
    if retries > 5:
        logger.warning(f"Warning: No questions were generated by this qa_tuple after 5 retries in generate_questions function. Returning None.")
        return None

    logger.info(f"The 'generate_questions' function for current tuple is complete. Appending data to 'questions' list variable...")
    for match in matches:
        questions.append(
            (
            match[0].replace(") ", "", 1).strip(),
            match[1].replace(") ", "", 1).strip(),
            para_tuple[0].replace(") ", "", 1),
            para_tuple[1].replace(") ", "", 1),
            )
        )

    logger.info("Question data appended successfully. Returning completion and questions for current qa_tuple.")
    return questions, completion

def generate_questions_plan(para: str, LLM: dict):
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    """
    initialized_model = LLM['llm']
    prompt_content = {
        "text": para,
    }

    # Determine which paragraphs are worthy of making questions from
    # Analyze-Realize-Create-Example loop
    #logger.info(f"Function generate_questions_plan initiated.")

    # Load the prompt and the grammar.
    try:
        cot_prompt, question_plan_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "question_plan_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in 'generate_questions_plan' function while trying to import its prompt and grammar: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_generate_questions_plan_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_llm_presets') is True:
            logger.info("Overriding default LLM presets for 'generate_questions_plan' function.")
            initialized_model = overrides['llm']
            LLM['override_llm_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            cot_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            logger.info("Overriding the prompt for 'generate_questions_plan' function.")

        # Override the grammar if it's requested.
        if overrides.get('grammar'): 
            question_plan_grammar = overrides['grammar']
            logger.info("Overriding the grammar for 'generate_questions_plan' function.")

    except KeyError:
        logger.info("Overrides for 'generate_questions_plan' function not present. Using default presets.")


    start_time = time.time()
    logger.info("Generating 'generate_questions_plan' completion for paragraph...")

    if LLM['override_llm_presets'] is True:
        completion = initialized_model(
            cot_prompt, 
            grammar=question_plan_grammar
        )["choices"][0]["text"]
    else:
        completion = initialized_model(
            cot_prompt,
            max_tokens=8000,
            stop=["</s>", "# Input:"],
            echo=True,
            grammar=question_plan_grammar,
            temperature=0.8,  # min p settings, too inconsistent
            top_k=0,
            top_p=1,
            min_p=0.5,
        )["choices"][0]["text"]

    end_time = time.time()
    logger.info(f"Completion for 'generate_questions_plan' function generated in {(end_time - start_time) / 60} minutes. Extracting response...")

    # Extract plan
    response_pattern = re.compile(
        r"Reasoning and thought process \(being careful to only plan questions that are entirely based on the text provided\):([\s\S]*)", 
        re.IGNORECASE | re.DOTALL,
    )

    try:
        generation = response_pattern.search(completion).group(1)
        logger.info(f"\nResponse for 'generate_questions_plan' function ***\n{generation}\n*** Response for 'generate_questions_plan' function ***\n")
    except Exception as e:
        logger.exception(f"An Exception occured when searching for the response pattern in the LLM completion: {e}")

    return generation, completion

def strip_steps(instruction_text: str) -> str:
    """
    This function takes a string containing step-by-step instructions and removes the "Step N." prefix from each line.

    Parameters:
    instruction_text (str): A string with each step in the format "Step N. Instruction", separated by newlines

    Returns:
    str: A single string with the steps stripped, joined by newlines.
    """
    instructions = instruction_text.split("\n")
    stripped_instructions = []

    for line in instructions:
        # Check if line starts with 'Step' and followed by a number and period
        if line.strip().startswith("Step") and "." in line:
            # Find the index of the first period
            period_index = line.find(".")
            # Extract the text after the period (and optional space)
            text_after_period = line[period_index + 1 :].lstrip()
            stripped_instructions.append(text_after_period)
        else:
            stripped_instructions.append(line)

    return "\n".join(stripped_instructions)

def vet_answer_accuracy_loop(qa_tuple: Tuple, total_retries: int, LLM: dict, run_id):
    try:
        qtuple = qa_tuple
        logger.info(f"\n\nStarting ACCURACY loop for question: \nqtuple: {qtuple[0]} \ncontext: {qtuple[2]}\n\n")

        passed_checks = 0
        times_checked = 0
        dissenting_reasoning = ""

        while times_checked < get_config("DOUBLE_CHECK_COUNTER"): # What the hell is this?
            logger.info(f"\n\nACCURACY CALL CHECK ANSWER: \nqtuple: {qtuple[0]} \ncontext: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}\n\n")

            judgement, answer_accuracy_output = check_answer(qtuple, LLM)
            write_output_to_file(answer_accuracy_output, "./check_answer_accuracy_generations", run_id)

            if not judgement[0]:  # if not accurate
                dissenting_reasoning = judgement[1]
            else:
                passed_checks += 1

            times_checked += 1
            if passed_checks >= ceil(get_config("DOUBLE_CHECK_COUNTER") / 2):
                break
            failed_checks = times_checked - passed_checks
            if failed_checks >= ceil(get_config("DOUBLE_CHECK_COUNTER") / 2):
                break

        if passed_checks >= ceil(get_config("DOUBLE_CHECK_COUNTER") / 2):  # if question checks passed
            logger.info(f"\n\nANSWER ACCURACY CHECKS PASSED \nretries: {total_retries}\n\n")
            return qtuple
        else:
            # Generate new question and restart the loop
            logger.info(f"\n\nACCURACY CHECKS FAILED - SENDING BACK TO QUESTION LOOP \nretries: {total_retries}\n\n")

            total_retries += 1
            qtuple, generate_new_q_output = generate_new_question(qtuple, LLM)
            write_output_to_file(generate_new_q_output, "./regenerate_question_generations", run_id)

            vet_question_loop(qtuple, LLM, total_retries, run_id=run_id.split("--subquestion--")[0])  # going to get one hell of a call stack by the end of this, but it should be fine

    except Exception as e:
        logger.exception(f"An Exception occurred for vet_answer_accuracy_loop function within generate_qa_tuples function: {e}")
        pass

    return (None, None, None, qtuple[3])

def vet_answer_relevance_loop(qa_tuple: Tuple, LLM: dict, total_retries: int, run_id):
    try:
        qtuple = qa_tuple
        logger.info(f"\n\nStarting RELEVANCE loop for question: \nqutple: {qtuple[0]} \ncontext: {qtuple[2]}\n\n")

        passed_checks = 0
        times_checked = 0
        dissenting_reasoning = ""

        while times_checked < get_config("DOUBLE_CHECK_COUNTER"):
            logger.info(f"\n\nRELEVANCE CALL CHECK ANSWER: \nqtuple: {qtuple[0]} \ncontext: {qtuple[2]} \nretries: {total_retries} \ndissenting reasoning: {dissenting_reasoning}\n\n")

            judgement, answer_relevancy_output = check_answer_relevancy_with_text(qtuple, LLM)
            write_output_to_file(answer_relevancy_output, "./check_answer_relevancy_generations", run_id)

            if not judgement[0]:  # if not relevant
                dissenting_reasoning = judgement[1]
            else:
                passed_checks += 1
            times_checked += 1

            if passed_checks >= ceil(get_config("DOUBLE_CHECK_COUNTER") / 2):
                break

            failed_checks = times_checked - passed_checks

            if failed_checks >= ceil(get_config("DOUBLE_CHECK_COUNTER") / 2):
                break

        if passed_checks >= ceil(get_config("DOUBLE_CHECK_COUNTER") / 2):
            logger.info(f"\n\nRELEVANCE CHECKS PASSED\n\n")

            return vet_answer_accuracy_loop(qtuple, total_retries, LLM, run_id)
        else:
            logger.info(f"\n\nRELEVANCE CHECKS FAILED - SENDING BACK TO QUESTION LOOP\n\n")

            total_retries += 1
            qtuple, generate_new_q_output = generate_new_question(qtuple, LLM)
            write_output_to_file(generate_new_q_output, "./regenerate_question_generations", run_id)

            return vet_question_loop(qtuple, LLM, total_retries, run_id=run_id.split("--subquestion--")[0])

    except Exception as e:
        logger.exception(f"An Exception occurred for vet_answer_relevance_loop function within generate_qa_tuples function: {e}")
        pass

    return (None, None, None, qtuple[3])

def vet_question_loop(qa_tuple: Tuple, LLM: dict, total_retries: int, run_id=None):
    try:
        question_group_id = run_id # Hacky...
        qtuple = qa_tuple
        logger.info(f"\n\nStarting QUESTION loop for question: n\qtuple:{qtuple[0]}, context: {qtuple[2]}\n\n")

        while total_retries <= 4:
            run_id = question_group_id + "--subquestion--" + make_id()
            passed_checks = 0
            times_checked = 0
            dissenting_reasoning = ""

            while times_checked < get_config("DOUBLE_CHECK_COUNTER"):
                logger.info(f"\n\nQUESTION CALL CHECK ANSWER: \n{qtuple[0]}\ncontext: {qtuple[2]}\nretries: {total_retries} \ndissenting reasoning: {dissenting_reasoning}\n\n")

                judgement, check_q_output = check_question(qtuple, LLM)
                write_output_to_file(check_q_output, "./check_question_generations", run_id)

                if not judgement[0]:  # if not relevant
                    dissenting_reasoning = judgement[1]
                else:
                    passed_checks += 1

                times_checked += 1
                if passed_checks >= ceil(get_config("DOUBLE_CHECK_COUNTER") / 2):
                    break
                failed_checks = times_checked - passed_checks
                if failed_checks >= ceil(get_config("DOUBLE_CHECK_COUNTER") / 2):
                    break

            if passed_checks >= ceil(get_config("DOUBLE_CHECK_COUNTER") / 2):  # if all question checks passed
                logger.info(f"\n\nQUESTION CHECKS PASSED retries: {total_retries}")
                return vet_answer_relevance_loop(qtuple, LLM, total_retries, run_id)
            else:
                # Generate new question and restart the loop
                logger.info(f"\n\nQUESTION CHECKS FAILED - GENERATING NEW QUESTION \nretries: {total_retries}\n\n")
                total_retries += 1

                if (total_retries <= 4):  # only regen question if we're not already at max regens
                    qtuple, generate_new_q_output = generate_new_question(qtuple, LLM)
                    write_output_to_file(generate_new_q_output,"./regenerate_question_generations",run_id,)
                    logger.info("New question: ", qtuple)
                # no calling of vet_question_loop, since we're already in a while loop

    except Exception as e:
        logger.exception(f"An Exception occurred in vet_question_loop function within generate_qa_tuples function: {e}")

    return (None, None, None, qtuple[3])


class GenerateQATuplesSimple: #TODO Write function documentation. This class will be a BEAR to create, and will likely need to be split up into separate nodes later on depending on its complexity.
    """
    This node generates QA tuples from input paragraphs using the 'generate_qa_tuples' function.
    This one has its presets hard-coded in, except for total retries.

    :param filtered_worthy_for_questions: Filtered Output from the JudgeParagraphs node. Default format: tuples of a paragraph chunk and source meta-data.
    :param LLM: An initialized model.
    :param qa_tuple_directory_name: The name of directory where the QA tuples will go.
    :param total_retries: The total number of retries the QA tuples should be cross-checked before being saved to output.
    :return vetted_qa_tuples: A list of QA tuples, vetted for relevance and accuracy.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LLM": ("LLM",),
                "total_retries": ("INT", {"default": 3, "min": 0, "max": 10000, "step":1}), # TODO: Maybe add dynamic total_retries option, so that it cuts off after a threshold of good-to-bad tuples is reached?
            },
            "optional": {
                "filtered_worthy_for_questions": ("TUPLE", {"forceInput": True}),
            },
        }
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("vetted_qa_tuples",)

    FUNCTION = "return_generate_qa_tuples"

    CATEGORY = "augmentoolkit_functions"

    def return_generate_qa_tuples(self, LLM, total_retries, filtered_worthy_for_questions=None):
        # Get the node starttime.
        node_start_time = time.time()
        
        # Initialize the presets.
        qa_tuple_directory_name = "qatuples_raw"
        question_plan_directory_name = "question_plan_generations"
        question_plan_generations_directory_name = "question_generation_generations"
        
        
        # Run the generate_qa_tuples function 
        vetted_qa_tuples = generate_qa_tuples(LLM, 
                                              qa_tuple_directory_name, 
                                              question_plan_directory_name, 
                                              question_plan_generations_directory_name, 
                                              total_retries, 
                                              filtered_worthy_for_questions)

        node_end_time = time.time()
        logger.info(f"\nGenerate QA Tuples (Simple) node complete! \nTotal Runtime: {(node_end_time - node_start_time) / 60} minutes.")

        return(vetted_qa_tuples,)

#TODO Write function documentation. This class will be a BEAR to create, and will likely need to be split up into separate nodes later on depending on its complexity.
class GenerateQATuplesAdvanced: 
    """
    This node generates QA tuples from input paragraphs using the 'generate_qa_tuples' function.
    This one has options to customize the directory names and override the LLM presets.

    :param filtered_worthy_for_questions: Filtered Output from the JudgeParagraphs node. Default format: tuples of a paragraph chunk and source meta-data.
    :param LLM: An initialized model and its presets.
    :param qa_tuple_directory_name: The name of directory where the QA tuples will go.
    :param total_retries: The total number of retries the QA tuples should be cross-checked before being saved to output.
    :return vetted_qa_tuples: A list of QA tuples, vetted for relevance and accuracy.
    """

    @staticmethod
    def update_llm_with_override(LLM, override_key, override_value):
        if override_value is not None:
            LLM[override_key] = override_value

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LLM": ("LLM",),
                "qa_tuple_directory_name": ("STRING", {"default": 'qatuples_raw'}),
                "question_plan_directory_name": ("STRING", {"default": 'question_plan_generations'}),
                "question_plan_generations_directory_name": ("STRING", {"default": 'question_generation_generations'}),
                "total_retries": ("INT", {"default": 3, "min": 0, "max": 10000, "step":1}), # TODO: Maybe add dynamic total_retries option, so that it cuts off after a threshold of good-to-bad tuples is reached?
            },
            "optional": {
                "filtered_worthy_for_questions": ("TUPLE", {"forceInput": True}),
                # Since each node contains several prompts, grammars, and presets,
                # We need import to multiple LLM initializtion dictionaries.
                # This way, we can mix and match default settings with user-overridden ones.
                "override_check_answer_presets": ("LLM",),
                "override_check_answer_relevancy_with_text_presets": ("LLM",),
                "override_check_question_presets": ("LLM",),
                "override_generate_new_question_presets": ("LLM",),
                "override_generate_questions_plan_presets": ("LLM",),
                "override_generate_questions_presets": ("LLM",),
            },
        }
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("vetted_qa_tuples",)

    FUNCTION = "return_generate_qa_tuples"

    CATEGORY = "augmentoolkit_functions/advanced"

    def return_generate_qa_tuples(self, LLM, 
                                  qa_tuple_directory_name, 
                                  question_plan_directory_name, 
                                  question_plan_generations_directory_name, 
                                  total_retries, 
                                  filtered_worthy_for_questions=None,
                                  override_check_answer_presets=None,
                                  override_check_answer_relevancy_with_text_presets=None,
                                  override_check_question_presets=None,
                                  override_generate_new_question_presets=None,
                                  override_generate_questions_plan_presets=None,
                                  override_generate_questions_presets=None,
                                  ):
        # Get the node starttime.
        node_start_time = time.time()
        
        # Put the optional override LLM dictionaries into the main LLM dictionary object.
        # This way, we don't have to keep track of the overrides individually.
        
        self.update_llm_with_override(LLM, 'override_check_answer_presets', override_check_answer_presets)
        self.update_llm_with_override(LLM, 'override_check_answer_relevancy_with_text_presets', override_check_answer_relevancy_with_text_presets)
        self.update_llm_with_override(LLM, 'override_check_question_presets', override_check_question_presets)
        self.update_llm_with_override(LLM, 'override_generate_new_question_presets', override_generate_new_question_presets)
        self.update_llm_with_override(LLM, 'override_generate_questions_plan_presets', override_generate_questions_plan_presets)
        self.update_llm_with_override(LLM, 'override_generate_questions_presets', override_generate_questions_presets)

        # Run the generate_qa_tuples function 
        vetted_qa_tuples = generate_qa_tuples(LLM, 
                                              qa_tuple_directory_name, 
                                              question_plan_directory_name, 
                                              question_plan_generations_directory_name, 
                                              total_retries, 
                                              filtered_worthy_for_questions)

        node_end_time = time.time()
        logger.info(f"\nGenerate QA Tuples (Advanced) node complete! \nTotal Runtime: {(node_end_time - node_start_time) / 60} minutes.")

        return(vetted_qa_tuples,)



##########################################################
#### MakeDatasetMultiturnConversationSimple Functions ####
##########################################################
"""
            tasks = [augmentoolkit_async_functions.create_info(
                idx,
                group,
                LLM, 
                ASSISTANT_MODE, 
                convs_info,
                multi_turn_convs_info_dir, 
                REARRANGEMENTS_TO_TAKE) for idx, group in enumerate(convs_info)]
            asyncio.run(run_tasks(tasks))


"""

def ensure_multiple_answers_are_same(info: str, conv: str, LLM: dict):  # why is this a whole separate function? Once upon a time, LLMs were used in validation here, too. But programmatic validation SEEMS to catch the common problems. This is here so that I can add it back in if I have to.
    """Loop to ensure that the answer is consistent in the conversation and in the tuple."""
    retries = 0
    c = conv

    while retries < 2:  # try twice, since multiturn is an expensive operation
        if call_all_processors(c[0], info[0]):  # if programmatic validation passes
            return c
        retries += 1

        if retries >= 2:
            return None

        # If we're here, majority of relevance checks failed
        logger.warning(f"\nMajority of relevance checks failed in 'ensure_multiple_answers_are_same' function. \n")
        print("----------------\n\n\n\nRETRYING!!!!\n\n\n\n----------------")
        # Broken info is 1) rare and 2) handled by the retry limit. We don't want to waste compute on regenerating info as they take time.
        retry = make_multiturn_conversation(info, LLM)

        if retry is not None:  # Note: retry CANNOT actually be None
            c = retry
        else:
            # If we failed to generate a retry, don't waste compute
            logger.info("'ensure_multiple_answers_are_same' function failed for this conversation! Returning None")
            return None

    return None

def make_multiturn_conversation(info, LLM: dict):

    conv, conv_output = multi_turn_conversation(
        info[0], info[1], info[2], info[3], 
        LLM, 
        assistant_mode=get_config("ASSISTANT_MODE") 
    )  # based on what was originally: multi_turn_conversation(qa_tuples, character, scenario, scenario_plan, initialized_model)

    write_output_to_file(conv_output, "./multiturn_conversation_generations", info[4])

    return conv

def make_regenerate_answer_constrain_to_text_plan(prompt: str, qatuple: Tuple, dissenting_reasoning: str, LLM: dict):
    retries = 0
    initialized_model = LLM['llm']
    prompt_content = {
        "dissenting_reasoning": strip_steps(dissenting_reasoning),
        "qatuple": qatuple,
    }

    # Load the prompt and the grammar.
    try:
        decision_prompt, answer_constrain_to_text_plan_grammar = load_external_prompt_and_grammar("make_regenerate_answer_constrain_to_text_plan", "answer_constrain_to_text_plan_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in 'make_regenerate_answer_constrain_to_text_plan' function while trying to import its prompt and grammar: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_make_regenerate_answer_constrain_to_text_plan_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_llm_presets') is True:
            logger.info("Overriding default LLM presets for 'make_regenerate_answer_constrain_to_text_plan' function.")
            initialized_model = overrides['llm']
            LLM['override_llm_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            question_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            logger.info("Overriding the prompt for 'make_regenerate_answer_constrain_to_text_plan' function.")

        # Override the grammar if it's requested.
        if overrides.get('grammar'): 
            questions_grammar = overrides['grammar']
            logger.info("Overriding the grammar for 'make_regenerate_answer_constrain_to_text_plan' function.")

    except KeyError:
        logger.info("Overrides for 'make_regenerate_answer_constrain_to_text_plan' function not present. Using default presets.")

    while retries < 5:
        try:
            start_time = time.time()
            logger.info(f"Generating 'make_regenerate_answer_constrain_to_text_plan' completion...")

            if LLM['override_llm_presets']:
                completion = initialized_model(
                    decision_prompt,
                    grammar=answer_constrain_to_text_plan_grammar
                )["choices"][0]["text"]
            else:
                completion = initialized_model(
                    decision_prompt,
                    max_tokens=3000,
                    stop=["</s>", "# Input:"],
                    echo=True,
                    grammar=answer_constrain_to_text_plan_grammar,
                    temperature=0.2,
                )["choices"][0]["text"]

            if get_config("DEBUG_MODE"):
                logger.info(f"\n*** make_regenerate_answer_constrain_to_text_plan COMPLETION ***: \n{completion}\n ***make_regenerate_answer_constrain_to_text_plan COMPLETION ***\n")

            end_time = time.time()
            logger.info(f"Done! Completion took {(end_time - start_time) / 60} minutes to generate.")
            logger.info(f"Completion for 'make_regenerate_answer_constrain_to_text_plan' function generated. Extracting correction...")

            completion_pattern = re.compile(
                r"Reasoning and thought process:\n(.+)", 
                re.DOTALL
            )

            correction = completion_pattern.search(completion).group(1)
            if get_config("DEBUG_MODE"):
                logger.info(f"\n*** make_regenerate_answer_constrain_to_text_plan CORRECTION ***: \n{correction}\n ***make_regenerate_answer_constrain_to_text_plan CORRECTION ***\n")
            logger.info(f"Correction extraction successful.")

            return correction

        except Exception as e:
            retries += 1
            logger.exception(f"An Exception occured with completion creation in 'make_regenerate_answer_constrain_to_text_plan' function: {e}\nHere's the completion:\n{completion}")

def multi_turn_conversation(qatuples: list, character: str, scenario: str, scenario_plan: str, LLM: dict, assistant_mode=False):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. 
    The character's personality and back story should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """
    charname = extract_name(character)
    first_words_of_card = extract_first_words(charname, character)
    conv_starters = [  # prevents it from regurgitating the card (when combined with filtering)
        "Ah",
        "Oh",
        # "You",
        # "Really",
        "I",
        # "What",
        # "So",
        "Welcome",
        "Hey",
        # "Look",
        # "Now",
        # "Huh",
        "It's",
        "Hello",
    ]

    conv_starters_filtered = [
        starter for starter in conv_starters if starter not in first_words_of_card
    ]
    conv_starter = random.choice(conv_starters_filtered)
    logger.info(f"--CONV STARTERS FILTERED--\n{conv_starters_filtered}")

    initialized_model = LLM['llm']

    # Create grammar based off of the number of questions
    # TODO: Externalize this grammar and modify it so that it is called by the format_external_text_like_f_string function.

    # Suppress grammar output to console.
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    multi_turn_conversation_grammar = LlamaGrammar.from_string(
        f"""

    # The root rule defines the structure of the dialogue
    root ::= [^\\n]+ "\\n" question-1 anything

    # Define constants acquired from code
    character-name ::= "{charname}"
    
    intro-statement ::= character-name ":" [^\\n]+
    
    # Statement by Secondary Character
    question-1 ::= [^\\n]+ ":" [^\\n]+
    
    # Statement by Primary Character
    
    anything ::= [^\\t]+

    """
    )

    # Restore stdout and stderr after the grammar is loaded.
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Load the correct prompts and contextual information depending on whether or not the conversation is in assistant mode.
    if assistant_mode:
        logger.info(f"multi_turn_conversation ASSISTANT MODE ACTIVE")

        # Load the content to put in the assistant prompt.
        character = "AI Assistant"
        scenario = "A conversation between a helpful AI Assistant, and a user."
        scenario_plan = "N/A"
        charname = "AI Assistant"
        prompt_content = {
            "character": character,
            "charname": charname,
            "conv_starter": conv_starter,
            "scenario": scenario,
            "scenario_plan": scenario_plan,
            "question_answer_list": format_qatuples(qatuples),
        }

        # Load the assistant prompt.
        # Since the grammar is hard-coded into the function, we don't need to externally load it.
        try:
            cot_prompt, _ = load_external_prompt_and_grammar("multi_turn_conversation_assistant_mode", "answer_constrain_to_text_plan_grammar", prompt_content)
        except Exception as e:
            logger.exception(f"An Exception occurred in 'multi_turn_conversation' function while trying to import its assistant prompt: {e}")

        # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
        try: 
            overrides = LLM['override_multi_turn_conversation_assistant_mode_presets']

            # Override the default function presets if it's requested.
            if overrides.get('override_llm_presets') is True:
                logger.info("Overriding default LLM presets for 'multi_turn_conversation' function.")
                initialized_model = overrides['llm']
                LLM['override_llm_presets'] = True

            # Override the prompt if it's requested.
            if overrides.get('prompt'):
                question_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
                logger.info("Overriding the prompt for 'multi_turn_conversation' function.")

            # Override the grammar if it's requested.
            if overrides.get('grammar'): 
                questions_grammar = overrides['grammar']
                logger.info("Overriding the grammar for 'multi_turn_conversation' function.")

        except KeyError:
            logger.info("Overrides for 'multi_turn_conversation' function not present. Using default presets.")
    else:
        # Load the content to put in the regular prompt.
        extra_info = extract_steps(scenario_plan)
        prompt_content = {
            "character": character,
            "charname": charname,
            "conv_starter": conv_starter,
            "extra_info": extra_info,
            "scenario": scenario,
            "scenario_plan": scenario_plan,
            "question_answer_list": format_qatuples(qatuples),
        }

        # Load the regular prompt.
        try:
            cot_prompt, _ = load_external_prompt_and_grammar("multi_turn_conversation", "answer_constrain_to_text_plan_grammar", prompt_content)
        except Exception as e:
            logger.exception(f"An Exception occurred in 'multi_turn_conversation' function while trying to import its prompt: {e}")
            
        # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
        try: 
            overrides = LLM['override_multi_turn_conversation_presets']

            # Override the default function presets if it's requested.
            if overrides.get('override_llm_presets') is True:
                logger.info("Overriding default LLM generation presets for 'multi_turn_conversation' function.")
                initialized_model = overrides['llm']
                LLM['override_llm_presets'] = True

            # Override the prompt if it's requested.
            if overrides.get('prompt'):
                question_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
                logger.info("Overriding the prompt for 'multi_turn_conversation' function.")

            # Override the grammar if it's requested.
            if overrides.get('grammar'): 
                questions_grammar = overrides['grammar']
                logger.info("Overriding the grammar for 'multi_turn_conversation' function.")

        except KeyError:
            logger.info("Overrides for 'multi_turn_conversation' function not present. Using default presets.")

    # NOTE: Very rarely, the first message of this conv will just be part of the character card, causing the conv to not make much sense. 
    # The cause of this is likely the fact that Elise quotes her character card in her first message. 
    # However, referencing the character card in this way also makes characters act as they are described, which is deemed advantageous enough that I am not changing this for now.
    # I get the sense that LLMs can learn relationships and connections between parts of the prompt, even if they're quite far apart, if you give them examples like this. 
    # It's fascinating to see how each part of the prompt has consequences -- sometimes unintended ones.

    # Note: performance degrades rapidly if you put more than one sentence in a pre-prompt parentheses thing
    try:
        start_time = time.time()
        logger.info(f"Generating 'multi_turn_conversation' completion...")

        if LLM['override_llm_presets']:
            completion = initialized_model(
                cot_prompt,
                grammar=multi_turn_conversation_grammar,
            )["choices"][0]["text"]
        else:
            completion = initialized_model(
                cot_prompt,
                max_tokens=8000,
                stop=["</s>", "# Input:", "## Information"],
                echo=True,
                grammar=multi_turn_conversation_grammar,
                temperature=0.5,
                top_k=0,
                top_p=1,
                min_p=0.6,
            )["choices"][0]["text"]

            end_time = time.time()
            logger.info(f"Done! Completion took {(end_time - start_time) / 60} minutes to generate.")
            logger.info(f"Completion for 'multi_turn_conversation' function generated. Extracting response pattern...")
            if get_config("DEBUG_MODE"):
                logger.info(f"\n*** multi_turn_conversation COMPLETION ***: \n{completion}\n ***multi_turn_conversation COMPLETION ***\n")

    except Exception as e:
        logger.exception(f"An Exception occured in 'multi_turn_conversation' function while generating its completion: {e}")

    # Extract plan
    response_pattern = re.compile(
        f"Conversation that answers the provided question \(be sure that you do not change the questions or answers themselves; {charname} will answer the questions, not ask them; the questions and answers provided should be copied word for word, and surrounded by compelling conversation\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )

    generation = response_pattern.search(completion).group(1)
    if get_config("DEBUG_MODE"):
        logger.info(f"\n*** multi_turn_conversation GENERATION:***\n\n-------------------\n\n {generation} \n*** multi_turn_conversation GENERATION: ***\n\n-------------------\n\n")

    # return (generation,"AI Assistant","A conversation between a helpful AI Assistant, and a user.","N/A",qatuples), completion

    return (generation, character, scenario, scenario_plan, qatuples), completion


class MakeDatasetMultiturnConversationSimple: # TODO Write function documentation. Write dynamic path to get multi_turn_convs_info_dir folder.
    """
    This function creates a dataset of multi-turn conversations based on input multi-turn conversation data.
    This is what augmentoolkit is about at the end of the day: Making QA RP datasets from books.
    :param multi_turn_convs_info: 
    :param multi_turn_convs_info_dir: 
    :param arrangements_to_take: How many of the possible permutations of tuples in a group to take and make multi-turn conversations out of.
    :return None: The output of this class/node is the final dataset.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @staticmethod
    def read_json_files_info(directory):
        # Create a list to hold the tuples
        tuple_list = []

       # Get all the .json files in the directory, sorted
        json_files = sorted([f for f in os.listdir(directory) if f.endswith(".json")])

        # Read each file and convert the contents
        for file in json_files:
            with open(os.path.join(directory, file), "r") as f:
                data = json.load(f)

                # Ensure the data is in the correct format before converting to tuple
                if (
                    isinstance(data, list)
                    and len(data) == 5
                    and isinstance(data[0], list)
                    and all(len(item) == 4 for item in data[0])
                    and all(isinstance(i, str) for i in data[1:])
                ):
                    tuple_list.append((data[0], data[1], data[2], data[3], data[4]))

        return tuple_list

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "LLM": ("LLM",),
                "multi_turn_convs_info_dir": ("STRING", {"default": 'multi_turn_convs_info_dir'},),
                "multi_turn_convs_dir_arg" : ("STRING", {"default": 'multi_turn_convs_dir'},),
                "read_from_external_json": (["True","False"],),
            },
            "optional": {
                "multi_turn_convs_info": ("OUTPUT_TEXT",),
                #"override_multi_turn_conversation_presets": ("LLM",),
                #"override_multi_turn_conversation_assistant_mode_presets": ("LLM",),
                #"make_regenerate_answer_constrain_to_text_plan": ("LLM",),
            },
        }
    RETURN_TYPES = ()
    FUNCTION = "make_dataset_multi_turn_conversation"

    OUTPUT_NODE = True

    CATEGORY = "augmentoolkit_functions"

    def make_dataset_multi_turn_conversation(self, 
                                             LLM, 
                                             multi_turn_convs_info_dir, 
                                             multi_turn_convs_dir_arg, 
                                             read_from_external_json, 
                                             multi_turn_convs_info=None):

        # Set up the multi-turn conversation and results list.
        multi_turn_convs = []
        results = list()

        # Set up the output directory
        multi_turn_convs_dir = f"./{multi_turn_convs_dir_arg}"
        # Make the output directory if it doesn't exist.
        if not os.path.exists(multi_turn_convs_dir):
            os.makedirs(multi_turn_convs_dir)

        # Option to load in the conversation info directly from the previous function instead of reading it from json files.
        # Some people generating datasets have access to serious compute. Why slow them down?
        if read_from_external_json == "True": 
            convs_info = self.read_json_files_info(multi_turn_convs_info_dir) 
        elif read_from_external_json == "False" and multi_turn_convs_info is not None:
            convs_info = multi_turn_convs_info
        else:
            logger.error(f"Could not load data into 'make_dataset_multi_turn_conversation' function in class MakeDatasetMultiturnConversationSimple.")
            print("This was likely caused because 'read_from_external_json' is False and 'multi_turn_convs_info' is not connected to an input node.")

        if LLM['type'] == "llamacpp":
            try:
                # For all the information and their indexes within the conversation information...
                for idx, info in enumerate(convs_info):
                    #Set the file path for the multi-turn conversation output.
                    file_path = os.path.join(multi_turn_convs_dir, f"conv_{idx}.json")

                    # Skip if the file already exists
                    if not os.path.exists(file_path):
                        # Make a multi-turn conversation out of the information.
                        conv = make_multiturn_conversation(info, LLM)

                        # Make sure the multiple answers in the conversation are the same.
                        final_conv = ensure_multiple_answers_are_same(info, conv, LLM)

                        # Save the final multi-turn conversation in json format to the output directory.
                        if final_conv is not None:
                            with open(file_path, "w") as file:
                                json.dump(final_conv, file, indent=4)

                        # Append the final conversation into the multi-turn conversation list
                        multi_turn_convs.append(final_conv)

                    else:
                        # Skip the files that were already created
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            multi_turn_convs.append(data)
                        logger.info(f"Skipped generating {file_path} as it already exists")
                        logger.info("Dataset created!")

                    results.append({
                        "filename": f"conv_{idx}.json",
                        "folder": file_path,
                        "type": self.type
                    })

            except Exception as e:
                logger.exception(f"An Exception occured in 'make_dataset_multi_turn_conversation' function in class MakeDatasetMultiturnConversation: {e}")

        elif LLM['type'] == "aphrodite":
            tasks = [
                aug_async.create_conversation(
                    idx,
                    info, 
                    LLM, 
                    multi_turn_convs, 
                    multi_turn_convs_dir, 
                    assistant_mode=get_config("ASSISTANT_MODE"),
                    logging_level=get_config("LOG_LEVEL")
                ) for idx, info in enumerate(convs_info)
            ]
            asyncio.run(limited_tasks(tasks))
            logger.info("Dataset created!")

            results.append({
                "filename": f"conv_{idx}.json",
                "size_of_dataset": len([f for f in os.listdir(f"{folder_paths.get_output_directory()}") if f.startswith("conv_") and f.endswith(".json")]),
                "folder": file_path,
                "type": self.type
            })

        elif LLM['type'] == "api":
            tasks = [
                aug_async.create_conversation(
                    idx, 
                    info,
                    LLM, 
                    multi_turn_convs, 
                    multi_turn_convs_dir, 
                    assistant_mode=get_config("ASSISTANT_MODE"), 
                    completion_mode=get_config("COMPLETION_MODE"), 
                    logging_level=get_config("LOG_LEVEL")
                ) for idx, info in enumerate(convs_info)
            ]
            asyncio.run(limited_tasks(tasks))
            logger.info("Dataset created!")

            results.append({
                "filename": f"conv_{idx}.json",
                "folder": file_path,
                "type": self.type
            })

        else:
            logger.error(f"ERROR: Invalid LLM type selected in 'make_dataset_multi_turn_conversation' function in class MakeDatasetMultiturnConversationSimple.")

        return { "ui": { "results": results } }



#########################


class ChunkParagraphs:
    """
    This function takes a plaintext (txt) file and chunks it into paragraphs.
    Note that these paragraphs are NOT directly pulled from the plaintext file, 
    but are instead created based on an arbitrary token length as determined by the sentencepiece tokenizer.

    :param raw_text: text from a plaintext (txt) file, loaded in from the InputTextLoaderSimple node. Should be in UTF-8 format. (external)
    :param tokenizer: sentencepiece tokenizer (original: SentencePiece). (external)
    :param max_token_length: The maximum token length for a chunk of sentences.
    :return cleaned_paragraphs: List of sentence chunks with source text information. Should be tuples.
    """
    def __init__(self):
        self.source_name = folder_paths.get_raw_text_name()

    @staticmethod
    def sentence_chunking_algorithm(source_name: str, sentences, tokenizer, max_token_length: int):
        try: # Initialize the lists 
            sentence_chunks_with_source = []
            current_chunk = []
            token_count = 0

            # For every sentence in the tokenized text...
            for sentence in tqdm(sentences, desc=f"Processing {source_name.replace('.txt', '')}"):
                # Get the token length of the sentence.
                sentence_token_count = len(tokenizer.encode(sentence))

                # Add the sentence onto the previous one if their collective tokens are under the max token length.
                # Else, join the text file metadata onto the end instead and reset.
                if token_count + sentence_token_count <= max_token_length:
                    current_chunk.append(sentence)
                    token_count += sentence_token_count
                else:
                    sentence_chunks_with_source.append((" ".join(current_chunk), source_name.replace(".txt", "")))
                    current_chunk = [sentence]
                    token_count = sentence_token_count

            # Add the last chunk if it exists
            if current_chunk:
                sentence_chunks_with_source.append((" ".join(current_chunk), source_name.replace(".txt", "")))

            return sentence_chunks_with_source

        except Exception as e:
            logger.exception(f"An Exception occured in sentence_chunking_algorithm function in class ChunkParagraphs: {e}")

    @staticmethod
    def replace_unicode_escapes(text):
        # This function will replace all occurrences of Unicode escape sequences
        # in the form of \uXXXX (where X is a hexadecimal digit) with the corresponding Unicode character.
        # TODO ChatGPT wrote this, and it needs to be checked to see if it actually works as intended.
        return re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)

    @staticmethod
    def fix_text(to_replace_arr, text_to_fix):
        try:
            fixed_text = text_to_fix
            for replace_pair in to_replace_arr:
                # Fix the text based pre-specified replacement patterns
                fixed_text = fixed_text.replace(replace_pair[0], replace_pair[1])

                # Replace Unicode escape sentences with their corresponding Unicode characters.
                fixed_text = ChunkParagraphs.replace_unicode_escapes(fixed_text)

            return fixed_text
        except Exception as e:
            logger.exception(f"An Exception occurred in fix_text function in class ChunkParagraphs: {e}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_text": ("RAW_TEXT",{"forceInput": True}),
                "sentencepiece_tokenizer_name": ("STRING", {"default": 'Gryphe/MythoMax-L2-13b'}),
                "max_token_length": ("INT", {"default": 400, "min": 15, "max": 10000, "step": 1}), # 15 is low-end for the average length of a sentence in English.
            }
        }
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("cleaned_paragraphs",)
    FUNCTION = "return_clean_paragraphs"

    CATEGORY = "augmentoolkit_functions"

    def return_clean_paragraphs(self, raw_text: str, sentencepiece_tokenizer_name: str, max_token_length: int):

        # Load in the sentencepiece tokenizer. 
        # Note: This is NOT an LLM tokenizer, but one that splits up sentences into their individual words.
        try:
            if sentencepiece_tokenizer_name is not None:
                tokenizer = AutoTokenizer.from_pretrained(fr"{sentencepiece_tokenizer_name}")
            else:
                logger.error(f"ERROR: No sentencepiece tokenizer specified. Defaulting to 'Gryphe/MythoMax-L2-13b'")
                tokenizer = AutoTokenizer.from_pretrained("Gryphe/MythoMax-L2-13b") #Hardcode in the tokenizer is no tokenizer is specified by the user.
        except Exception as e:
            logger.exception(f"An Exception occured in return_clean_paragraphs function in class ChunkParagraphs when loading the tokenizer: {e}")

        # Initialize the sentence_chunks list.
        sentence_chunks = []

        # Remove Gutenberg header and footer
        raw_text = re.sub(r"^.*?START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*$\n", "", raw_text, flags=re.MULTILINE,)
        raw_text = re.sub(r"^.*?END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*$\n", "", raw_text, flags=re.MULTILINE,)

        #Sentence tokenize the text (SentencePiece)
        tokenized_text = sent_tokenize(raw_text)

        # Chunk the paragraphs and put them into the list.
        sentence_chunks += self.sentence_chunking_algorithm(self.source_name, tokenized_text, tokenizer, max_token_length)

        #Initialize the conversions to clean up the text, namely remove the newline symbol, closing gaps, and removing UTF-8 import errors.
        #TODO: Maybe turn this into an external dictionary if it gets super long???
        conversions = [
            ("\n", " "), ("  ", " "), ("\ufeff",""),
            ("\u2014", "-"), ("\u2019", "’"), ("\u00e6", "ae")
        ]

        #Clean up the text in the sentences chunks.
        cleaned_paragraphs = [
            (self.fix_text(conversions, seq[0]), seq[1]) for seq in sentence_chunks
        ]

        print(len(cleaned_paragraphs)) 
        print(cleaned_paragraphs[1]) 

        return (cleaned_paragraphs,)


class ConvertDirectoryToList: # TODO Write function documentation.
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path_arg": ("STRING", {"default": 'multi_turn_convs'}),
            }
        }
    RETURN_TYPES = ("MASTER_OUTPUT_TEXT","SIMPLIFIED_OUTPUT_TEXT")
    FUNCTION = "convert_directory_to_list"

    output_node = True

    CATEGORY = "augmentoolkit_functions"

    def convert_directory_to_list(directory_path_arg: str):
        # Set up lists and paths.
        directory_path = f"./ComfyUI/output/{directory_path_arg}/"
        master_list = []
        simplified_list = []

        # For every file in the directory path...
        for filename in os.listdir(directory_path):
            # Open any json files in the folder and load their contents.
            if filename.endswith(".json"):
                filepath = os.path.join(directory_path, filename)

                with open(filepath, "r") as file:
                    data = json.load(file)

                    # Load in the json data 
                    if isinstance(data, list) and all(isinstance(item, (list, str)) for item in data):
                        master_list.append(data)

                        # Extract and process conversation
                        conversation, primary_char_desc = data[0], data[1]
                        primary_char_name = extract_name(primary_char_desc)
                        dialogues = extract_conversation(conversation)

                        # Convert to simplified format
                        simplified_conversations = [] 
                        for i, (charname, message) in enumerate(dialogues):  # Skipping the first message
                            from_person = ("human" if charname == primary_char_name else "gpt")
                            simplified_conversations.append({"from": from_person, "value": f"{charname}: {message}"})

                        if simplified_conversations:  # If there are any conversations
                            simplified_list.append({"conversations": simplified_conversations})

        # Write the master list to a new .jsonl file
        with open("master_list.jsonl", "w") as file:
            for item in master_list:
                file.write(json.dumps(item) + "\n")

        # Write the simplified data to a different .jsonl file
        with open("simplified_data.jsonl", "w") as file:
            for item in simplified_list:
                file.write(json.dumps(item) + "\n")

        logger.info("Conversion complete. Master list written to 'master_list.json'. Simplified data written to 'simplified_data.json'.")
        return (master_list, simplified_list,)


class DisplayOutputText: #TODO Write documentation for this function. And get it to print to the node block, if possible.
    """
    Modified from the DisplayString class in NodeGPT: https://github.com/xXAdonesXx/NodeGPT/blob/main/DisplayText.py
    :param output_text: Text strings of any kind (external).
    :return NONE: The output is the text displayed in the console.
    """
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_text": ("STRING", {"forceInput": True}),
            },
            "hidden": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "display_output_text"

    output_node = True

    CATEGORY = "augmentoolkit_functions"

    def display_output_text (self, output_text: str):

        output_text = output_text['STRING']
        print(output_text)

        return {"ui": {"text": output_text}}


# TODO Fix this class! And figure out what the hell 'lst' means.
"""
class FilterAndFlatten:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "read_from_external_json": (["True","False"])
            },
            "optional": {
                "master_output_text": ("MASTER_OUTPUT_TEXT",),
            },
        }
    RETURN_TYPES = ("OUTPUT_TEXT",)
    FUNCTION = "filter_and_flatten"

    CATEGORY = "helper_function"

    def filter_and_flatten(self, read_from_external_json, master_output_text=None):
        if read_from_external_json == "True":
            with open("./processed_master_list.json") as f:
                first = f.read()
                data = json.loads(first)

        if master_output_text is not None:
            data = master_output_text

        if read_from_external_json == "True" and master_output_text is None:
            logging.error("No data was input into filter_and_flatten function in class FilterAndFlatten.")
            print("Either read it from a JSON or connect it to an input node such as 'Create a Simplified List Copy of the Dataset'.")

        flat_list = []
        # ??????????????
        # Loop through each sublist in the main list
        for sublst in lst:
            # Check if the first element of the sublist is itself a list (subsublist1)
            if isinstance(sublst[0], list):
                # Extend the flat_list with the elements from subsublist1
                flat_list.extend(sublst[0])

        return(flat_list, {"ui": { "text": len(flat_list)}})
"""



#TODO Write documentation for this function.
#TODO Make 
class FilterAndGraph: 
    """
    This function takes a json file (?) and determines which paragraphs are worth of making questions from.

    :param judged_worthy_for_questions: judged paragraph tuples from the JudgeParagraphs node.
    :return filtered_worthy_for_questions: judged paragraph tuples, but with the 'None's removed.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "judged_worthy_for_questions": ("TUPLE", {"forceInput": True}),
                "plot_paragraphs": (["True","False"],),
            }
        }

    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("filtered_worthy_for_questions",)
    FUNCTION = "filter_and_graph"

    CATEGORY = "augmentoolkit_functions"
    
    def filter_and_graph(self, judged_worthy_for_questions, plot_paragraphs):

        filtered_worthy_for_questions = None
        source_counts = Counter()
        directory_path = os.path.join(self.output_dir,"worthy_for_questions")

        if judged_worthy_for_questions is None:
            logger.info("'filtered_worthy_for_questions' variable is None! Reading in data from json folder...")
            try:
                for filename in os.listdir(directory_path):
                # Open any json files in the folder and load their contents.
                    if filename.endswith(".json"):
                        filepath = os.path.join(directory_path, filename)

                        with open(filepath, "r") as file:
                            judged_worthy_for_questions = json.load(file)

                            # Unpack the json file.
                            paragraph = judged_worthy_for_questions['paragraph']
                            source = judged_worthy_for_questions['metadata']

                            # Count the occurrences of None and non-None for each source text.
                            if paragraph is None:
                                source_counts[source] = source_counts.get(source, [0, 0])
                                source_counts[source][0] += 1
                            else:
                                source_counts[source] = source_counts.get(source, [0, 0])
                                source_counts[source][1] += 1
            except Exception as e:
                logger.exception(f"An Exception occurred in filter_and_graph function under class FilterAndGraph while trying to load jsons from '{directory_path}': {e}")
        else:
            for paragraph, source in judged_worthy_for_questions:
                if paragraph is None:
                    source_counts[source] = source_counts.get(source, [0, 0])
                    source_counts[source][0] += 1
                else:
                    source_counts[source] = source_counts.get(source, [0, 0])
                    source_counts[source][1] += 1

        # Prepare data for the graph.
        labels = list(source_counts.keys())
        none_counts = [source_counts[source][0] for source in labels]
        logger.info(f"judged_worthy_for_questions none count: {none_counts}")
        non_none_counts = [source_counts[source][1] for source in labels]
        logger.info(f"judged_worthy_for_questions non-none count: {non_none_counts}")

        file = f"paragraphs_suitability_{make_id()}.png"

        # Plot the graph, then export it to outputs.
        x = range(len(labels))
        plt.bar(x, none_counts, width=0.4, label="Not suitable", align="center")
        plt.bar(x, non_none_counts, width=0.4, label="Valid Paragraphs", align="edge")
        plt.xlabel("Source Text")
        plt.ylabel("Number of Paragraphs")
        plt.title("Paragraphs Suitable for Questions by Source Text")
        plt.xticks(x, labels, rotation="vertical")
        plt.legend()
        #plt.tight_layout() Comfy throws a warning and skips over this part of the graph.
        plt.savefig(f"{self.output_dir}/{file}", dpi=300)

        paragraphs_suitability_plot = mpimg.imread(f"{self.output_dir}/{file}")

        # Filter out tuples with None and return the new list.
        filtered_worthy_for_questions  = [t for t in judged_worthy_for_questions if t[0] is not None]
        logger.info(f"filtered_worthy_for_questions: {filtered_worthy_for_questions[0]} : filtered_worthy_for_questions")

        if plot_paragraphs == "True":
            return (filtered_worthy_for_questions, { "ui": {"paragraphs_suitability_plot": paragraphs_suitability_plot} })
        else:
            return (filtered_worthy_for_questions,)

#This is NOT a terminal node, as the outputs of it could be useful for diagnostic purposes.
class ConvertDirectoryAndProcessConversations: #TODO Maybe merge into class ConvertDirectoryToList as a separate function?
    """
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path_arg": ("STRING", {"default": 'multi_turn_convs'}),
            }
        }
    RETURN_TYPES = ("MASTER_OUTPUT_TEXT",)
    FUNCTION = "convert_directory_to_list"

    CATEGORY = "augmentoolkit_functions"

    def convert_directory_to_list(self, directory_path_arg):
        directory_path = f"./{directory_path_arg}/"
        master_list = []

        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                filepath = os.path.join(directory_path, filename)

                with open(filepath, "r") as file:
                    data = json.load(file)

                    if isinstance(data, list) and all(isinstance(item, (list, str)) for item in data):
                        # Extract and process the conversation part
                        conversations = extract_conversation(data[0])
                        # Convert tuples back to the formatted string as required

                        data[0] = [
                            f"{charname}: {message}" for charname, message in conversations
                        ]
                        master_list.append(data)
                    else:
                        logger.error(f"File {filename} is not in the expected format.")

        # Write the master list to a new file
        with open("processed_master_list.json", "w") as file:
            json.dump(master_list, file)

        logger.info("Conversion complete. The processed master list is written to 'processed_master_list.json'.")
        return (master_list,)

###########################################
######## JudgeParagraphs Functions ########
###########################################


# Wrapper for the async 'filter_all_questions' function, since the main class function is not async.
# TODO Change augmentoolkit_async_2_functions to augmentoolkit_async_functions when testing is complete.
async def await_filter_all_questions(paragraphs_processed,
                                     judged_worthy_for_questions, 
                                     LLM: dict, 
                                     output_dir: str, 
                                     USE_SUBSET: bool, 
                                     use_filenames=None, 
                                     rtwl=run_task_with_limit):
    return await aug_async.filter_all_questions(paragraphs_processed, 
                                                judged_worthy_for_questions, 
                                                LLM, 
                                                output_dir, 
                                                take_subset=USE_SUBSET, 
                                                use_filenames=None,
                                                rtwl=rtwl,
                                                completion_mode=get_config("COMPLETION_MODE"), #Hardcode these for right now.
                                                logging_level=get_config("LOG_LEVEL")
                                                )


def filter_all_questions(cleaned_paragraphs: str, judged_worthy_for_questions, LLM: dict, worthy_for_questions_output_dir: str):

    # For each index and paragraph in the cleaned text...
    for idx, paragraph in tqdm(enumerate(cleaned_paragraphs)):
    # Get the file name and path. Note that in this case the file MUST be a json.
        file_name = f"{idx}.json"
        file_path = os.path.join(worthy_for_questions_output_dir, file_name)

        # Check if the judgment for this paragraph already exists
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
                logger.info(f"LOADING: {data}")

                # If the data is a string, don't do anything.
                # If it isn't, append the paragraph and meta-data.
                if isinstance(data, str):
                    judged_worthy_for_questions.append((None, data[7:]))
                else:
                    judged_worthy_for_questions.append((data["paragraph"], data["metadata"]))

        else:
            # If the file doesn't exist, judge the input paragraph's quality for generating QA questions.
            judgment = judge_paragraph(paragraph, LLM)
            logger.info(f"LLM judgement completion: {judgment}")
            judged_worthy_for_questions.append(judgment)

            # Prepare the data to be written to the file
            if judgment[0] is not None:
                # The paragraph passed the judgment
                data_to_write = {"paragraph": judgment[0], "metadata": judgment[1]}
            else:
                # The paragraph did not pass the judgment
                data_to_write = f"failed|{judgment[1]}"

                # Write the judgment to a unique file as JSON
                with open(file_path, "w") as file:
                    json.dump(data_to_write, file)

            # Debug messages
            try:
                if judgment[0] is not None:
                    logger.info(f"DEBUG model decided that index {idx} was suitable")
                else:
                    logger.warning(f"DEBUG model decided that index {idx} was not suitable")
            except:
                logger.exception(f"DEBUG max retries exceeded for index {idx}")

    return judged_worthy_for_questions


def judge_paragraph(p: str, LLM: dict):
    # Initialize variables
    initialized_model = LLM['llm']
    reached_decision = False
    max_retries = 0
    prompt_content = {
        "p": p,
    }

    # Load the prompt and grammar.
    try:
        decision_prompt, judge_paragraph_grammar = load_external_prompt_and_grammar("judge_paragraph","judge_paragraph_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in 'judge_paragraph' function while trying to import its prompt and grammar: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_judge_paragraph_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_llm_presets') is True:
            logger.info("Overriding default LLM presets for 'judge_paragraph' function.")
            initialized_model = overrides['llm']
            LLM['override_llm_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            question_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            logger.info("Overriding the prompt for 'judge_paragraph' function.")

        # Override the grammar if it's requested.
        if overrides.get('grammar'): 
            question_grammar = overrides['grammar']
            logger.info("Overriding the grammar for 'judge_paragraph' function.")

    except KeyError:
        logger.info("Overrides for 'judge_paragraph' function not present. Using default presets.")

    logger.info(f"\nParagraph being judged: \n{p} \ntype: {type(p)}")
    time.sleep(5)

    while not reached_decision and (max_retries <= 3):

        # Load the initialized LLM and judge the paragraph.
        try:
            start_time = time.time()
            logger.info(f"Generating 'judge_paragraph' completion... \nCurrent Retry Count: {max_retries}")

            # Set up the option to override the functions generation presets
            if LLM['overide_llm_presets']:
                completion = initialized_model(
                        decision_prompt, 
                        grammar=judge_paragraph_grammar
                )["choices"][0]["text"]
            else:
                completion = initialized_model(
                    decision_prompt,
                    max_tokens=6000,
                    grammar=judge_paragraph_grammar,
                    stop=["</s>", "# Input:"],
                    echo=True,
                    temperature=0.2,
                )["choices"][0]["text"]

            end_time = time.time()
            logger.info(f"Completion took {(end_time - start_time) / 60} minutes to generate.")
            logger.info(f"Completion for 'judge_paragraph' function on retry {max_retries} generated. Extracting response pattern...")

        except Exception as e:
            logger.exception(f"An Exception occured in 'judge_paragraph' function in class JudgeParagraphs while trying to generate an LLM completion: {e}")
            break

        response_pattern = re.compile(
            r"Reasoning and thought process \(reason intelligently\):(.+)",
            re.DOTALL | re.IGNORECASE,
        )
        judgement_pattern = re.compile(
            r"Final Judgment:(.+)",
            re.DOTALL | re.IGNORECASE
        )

        # Extract the response pattern and determination from the completion.
        try:
            response = response_pattern.search(completion).group(1)
            if get_config("DEBUG_MODE"):
                logger.info(f"\n*** Response for 'judge_paragraph' function ***\n{response}\n*** Response for 'judge_paragraph' function ***\n")

            print("-------------------")

            determination = judgement_pattern.search(response).group(1)
            logger.info(f"\n\nDETERMINATION:\n------\n{determination}\n---------\n")

            if "unsuitable" in determination.lower():
                reached_decision = True
                return (None, p[1])
            elif "suitable" in determination.lower():
                return (p[0], p[1])

        except Exception as e:
            logger.exception(f"An exception occured in judge_paragraph function under class JudgeParagraphs: {e}")
            break

        max_retries += 1


# TODO Create the API route.
# TODO Make worthy_for_questions_output_dir editable by the user, although doing so might break things.
# TODO: Actually rename all the variables, for consistancy's sake.
class JudgeParagraphs:
    """
    This node takes a tuple of chunked paragraphs with source meta-data and determines which paragraphs are worth of making questions from.

    :param cleaned_text: chunked paragraph tuples, either from the ChunkParagraphs node or a previously-made json (external).
    :param LLM: An initialized LLM and its presets (external).
    :param text_manually_cleaned: Load the chunked paragraphs from a previously-made json instead of the ChunkParagraphs node.
    :param max_token_length: The maximum token length for a chunk of sentences.
    :return judged_worthy_for_questions: List of paragraphs deemed worthy of generating QA tuples from.
    """
    USE_SUBSET = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cleaned_paragraphs": ("TUPLE", {"forceInput": True}),
                "LLM": ("LLM",),
                "text_manually_cleaned_arg": (["False", "True"],),
            },
            "optional": {
                "override_judge_paragraph_presets": ("LLM",),
            }
        }

    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("judged_worthy_for_questions",)
    FUNCTION = "return_judged_worthy_for_questions"

    CATEGORY = "augmentoolkit_functions"

    def return_judged_worthy_for_questions(self, cleaned_paragraphs, LLM, text_manually_cleaned_arg, override_judge_paragraph_presets=None):

        if get_config("DEBUG_MODE"):
            for category, setting in LLM.items():
                logger.info(f"LLM Configs\ncategory:{category}\nsetting{setting}")

        # Put the overrides in the LLM dictionary. Defaults to None per the function inputs.
        LLM['override_judge_paragraph_presets'] = override_judge_paragraph_presets
        llm_type = LLM.get('type')

        # Define the worthy_for_questions output directory.
        # Hardcode for right now
        worthy_for_questions_output_dir = "./ComfyUI/output/worthy_for_questions"

        # If the worthy_for_questions output directory doesn't exist, create it.
        if not os.path.exists(worthy_for_questions_output_dir):
            os.makedirs(worthy_for_questions_output_dir, exist_ok=True)

        if text_manually_cleaned_arg == "False":
            # Create the question container if the text hasn't been manually cleaned.
            judged_worthy_for_questions = []

            if llm_type == "llamacpp": # If the llm type is llama, send it down the regular Llama-cpp route.
                judged_worthy_for_questions = filter_all_questions(cleaned_paragraphs, judged_worthy_for_questions, LLM, )
                return(judged_worthy_for_questions,)

            elif llm_type == "aphrodite": # If the llm type is aphrodite, send it down the async aphrodite route.
                try:
                    judged_worthy_for_questions = asyncio.run(
                        await_filter_all_questions(
                            cleaned_paragraphs, 
                            judged_worthy_for_questions, 
                            LLM, 
                            worthy_for_questions_output_dir, 
                            get_config("USE_SUBSET")
                        )
                    )
                    return(judged_worthy_for_questions,)
                except RuntimeError as e:
                    logger.error(f"RUNTIME ERROR in aphrodite-engine route of class JudgeParagraphs: {e}")

            elif llm_type == "api": 
                # If the llm type is an api, send it down the api route. 
                # Since API and aphrodite are handled in the EngineWrapper class, the same code should work for both.
                try:
                    judged_worthy_for_questions = asyncio.run(
                        await_filter_all_questions(
                            cleaned_paragraphs, 
                            judged_worthy_for_questions, 
                            LLM, 
                            worthy_for_questions_output_dir, 
                            get_config("USE_SUBSET")
                        )
                    )
                    return(judged_worthy_for_questions,)
                except RuntimeError as e:
                    logger.error(f"RUNTIME ERROR in api route of class JudgeParagraphs: {e}")

            else: # Throw an error if none of these are the llm type.
                logger.error("VALUE ERROR in class JudgeParagraphs: An unsupported LLM type was input.")
                raise ValueError

        else:
            # No need to write to file since paragraph chunking is deterministic.
            judged_worthy_for_questions = cleaned_paragraphs

            return(judged_worthy_for_questions,)





###################################################
#### ReturnMultiturnConversationInfo Functions #### TODO: Make the file folders editable.
###################################################

def combine_traits(personality_matrix):  # GPT-generated

    # Using itertools.product to generate all possible combinations
    combinations = itertools.product(*personality_matrix)

    # Joining each combination into a single string
    combined_traits = [
        "\n".join(combination).strip().replace("\n\n", "\n")
        for combination in combinations
    ]

    return combined_traits

def special_instructions(n=1, non_axis_traits=False, non_axis_traits_only=False):
    """
    documentation todo
    """

    ### NOTE on how traits are planned out for this step ###
    # Here're the copy-pasted thoughts from my planning document, now slightly cleaned-up for the release of Augmentoolkit. The TLDR is at the bottom. The inspiration for this personality system is the main thing I gained from my math class this semester.
    # CHARACTER PLANNING
    # Consider that we can represent a character's personality a vector with multiple dimensions. Now, we could define any number of individual dimensions, and lots of them would be right: intelligence, extraversion, industriousness, etc. But in the default version of the Augmentool we're doing roleplay, so we want to pick a set of dimensions using which we can describe accurately and concisely the characters that might show up in a roleplay. Consider that if a personality trait is a vector in 3-space, we want to pick traits that aren't coplanar -- ie, that each describe something unique, though possibly with some partial overlap. Ideally, they'd all be perpendicular -- maximally unique traits.
    # I believe I have found 3 such axes that are useful for roleplay:
    # Assertiveness
    # Kindness/Morality
    # Horniness (one of the few things we have an edge over GPT in)
    # So we have
    # Chaste------------------------------------normal----------------------------------------------------------------Slaanesh
    # Shy/Withdrawn/Timid (Bocchi)--------------Has moments of shyness and courage------------------------------------James Bond
    # Kind--------------------------------------Good and bad sides ---------------------------------------------------politician
    # We make more verbose descriptions of each trait and place them in a matrix, reflecting the visualization above. We then create a list of all possible combinations of one item from each row and randomly sample from it for the special instruction.

    # NOTE TLDR In laymans terms: we make a grid of traits, where each row represents a trait and values along it indicate different expressions of that trait; then we pick one value from each row and shove it onto the context window as a "special instruction".

    # Two additional dimensions I thought of afterwards but have never tested: intellectual sophistication, and age. I might add these if testing shows that the AI can handle them, but no few-shot example has anywhere near 5 combinations, so we'll see.

    ## NOTE You may (and are encouraged to!) add your own trait dimensions here, to make the character personalities used more accurately reflect your specific use case and preference. Since every possible combination of one trait from each row is put into the list, you will get a lot of variety with your characters for not much work.
    # NOTE Chaste and puritan characters have a tendency to be interpreted by the AI as being religious, possibly because of "puritan", even though I initially just meant for this to be the opposite of horny. I'm leaving this in as a way to counteract occasional anti-religious bias and the AI's own personality.

    axis_traits = [
        [
            "The character should be chaste and puritanical.",
            "",
            "The character should be very seductive and flirtatious.",
        ],  # Horniness (middle deliberately left blank so that the model does not mention it, since "normal" people don't usually bring up sex in common conversation... right?)
        [
            "The character should be shy, withdrawn, and timid.",
            "The character should be neither particularly bold, nor particularly timid.",
            "The character should be assertive and bold.",
        ],  # Assertiveness
        [
            "The character should be kind and agreeable.",
            "The character should have both good and bad sides.",
            "The character should be an awful person, and should be enjoying every second of it."
            # "The character should be an awful person, possessing a number of vices (that are compatible with the previously-mentioned instructions)."
        ],  # Kindness/Morality
        # ["The character should be a young adult.", "the character should be middle-aged." "The character should be in late adulthood."], # Age group
        # ["The character should be unsophisticated and crude.", "The character should be decently smart and refined.", "The character should be the epitome of intellectual sophistication."],
    ]

    non_axis_trait_list = [  # The following are examples of traits that are not on the axes above, but are still useful for character creation. Typically use these if you want to easily hardcode your characters to all have a trait. I've not tested all of them, and I've not tested them in combination with the axis traits. But if you prefer a more manual approach to character creation, you can use stuff like this.
        """The character should be a catgirl who inserts "nya" into every sentence. and makes cat puns.""",  # someone actually has to do this, I'm serious, it'll be purrfect, nya~
        # They can be short and used in combination with the axis traits; or long and replace them.
        """The character should be a Japanese High School student.
The character should be a girl.
The character should be decently smart, but not genius-level.
The character should be very kind, but too gentle and too much of a pushover for their own good.""",
        """The character should be an awful person, and enjoying every second of it.
The character should be intellectually brilliant.
The character should be condescending and rude.""",
        """The character should be a young adult.
The character should be antisocial and coarse.
The character should be a smoker."""
        """The character should be middle-aged.
The character should be narcissistic."""
        # """The character should be edgy and nihilistic."""
    ]

    if not non_axis_traits_only:
        traits = combine_traits(axis_traits)

        selected_traits = random.sample(traits, 1)
        if non_axis_traits:
            selected_traits += random.sample(non_axis_trait_list, 1)

    if non_axis_traits_only:
        selected_traits = random.sample(non_axis_trait_list, 1)

    # Return the combined string, with each sentence on a new line
    return selected_traits[0]


def create_character_card_many_tuples(qatuples, plan, instructions, initialized_model, cheap_mode=False):  # Use cheap mode if you don't have the compute power to crank up the context to 8k using RoPE
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. 
    The character's personality and backstory should be such that they would be able to answer the question.
    Format: Question: [question]\n\n
    """

    # Consider appending part of the char card plan to the char card itself. Everything after the first period? It's really good material, be a shame to waste it.
    # This little thing is a hack to prevent the model from using the author of the given book as the character, which it was very insistent on doing
    author_name_letters = extract_capital_letters(qatuples[0][3])
    starting_str = ""
    exclusions = ["X", "Z", "Y", "Q"]
    if author_name_letters:
        starting_str = select_random_capital(exclusions + author_name_letters)
    else:
        starting_str = select_random_capital(exclusions)

    prompt_content = {
        "qatuples": qatuples,
        "instructions": instructions,
        "plan": plan,
        "starting_str": starting_str,
    }

    # Load the prompt and grammar.
    try:
        cot_prompt, character_card_grammar = load_external_prompt_and_grammar("create_character_card_many_tuples", "character_card_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in 'create_character_card_many_tuples' function while trying to import its prompt and grammar: {e}")

    try:
        start_time = time.time()
        #logger.info(f"create_character_card_many_tuples cot_prompt: {cot_prompt} : create_character_card_many_tuples cot_prompt")
        completion = initialized_model(
            cot_prompt,
            max_tokens=10000,
            stop=["</s>", "# Input:"],
            echo=True,
            grammar=character_card_grammar,
            #  temperature=0.2
            temperature=2,
            top_k=0,
            top_p=1,
            min_p=0.5,
        )["choices"][0]["text"]

        end_time = time.time()
        if get_config("DEBUG_MODE"):
            logger.info(f"\n*** Completion for 'create_character_card_plan_many_tuples' function ***\n{completion}\n*** Completion for 'create_character_card_plan_many_tuples' function ***\n")
        logger.info(f"Completion took {(end_time - start_time) / 60} minutes to complete.")
        logger.info(f"Completion for 'create_character_card_many_tuples' function generated. Extracting response pattern...")

    except Exception as e:
        logger.exception(f"An Exception occured in 'create_character_card_many_tuples' function: {e}")

    # Extract plan
    response_pattern = re.compile(
        r"Character card \(be creative, write at least 3 paragraphs for each dialogue line\):([\s\S]*)", 
        re.IGNORECASE | re.DOTALL,
    )
    generation = response_pattern.search(completion).group(1)
    if get_config("DEBUG_MODE"):
        logger.info(f"\nGeneration for 'create_character_card_many_tuples' function ***\n{generation}\n *** Generation for 'create_character_card_many_tuples' function")

    return generation, completion


# IMPORTANT FUNCTION.
def create_character_card_plan_many_tuples(qatuples, initialized_model):

    choose_which_special_instructions_function = [
        "special_instructions",
        "special_instructions_prototype",
        "special_instructions_prototype_2",
    ]

    #logger.info(f"\n*** create_character_card_plan_many_tuples function qatuples input ***\nqatuples:{qatuples}\n*** create_character_card_plan_many_tuples function qatuples input ***\n")
    #time.sleep(5)

    # Allow us to choose which special instructions to invoke in the code. Hard-coded for now.
    # TODO Make this a global variable, perhaps? - KR
    toggle_for_special_functions = "special_instructions"
    # if choose_which_special_instructions_function == "special_instructions"
    if toggle_for_special_functions == "special_instructions":
        instructions_string = special_instructions(n=1)
    elif toggle_for_special_functions == "special_instructions_prototype":
        #instructions_string = special_instructions_prototype(n=1)
        pass
    else:
        #instructions_string = special_instructions_prototype2(n=1)
        pass

    prompt_content = {
        "qatuples": qatuples,
        "instructions_string_strip": instructions_string.strip(),
        "format_qatuples_qatuples": format_qatuples(qatuples),
    }

    # Load the prompt and the grammar.
    try:
        cot_prompt, character_card_plan_grammar = load_external_prompt_and_grammar("create_character_card_plan_many_tuples", "character_card_plan_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in 'create_character_card_plan_many_tuples' function while trying to import its prompt and grammar: {e}")

    # Load the initialized LLM and create a character card plan.
    try:
        start_time = time.time()
        logger.info(f"Generating 'create_character_card_plan_many_tuples' completion...")

        completion = initialized_model(
            cot_prompt,
            max_tokens=8000,
            stop=["</s>", "# Input:"],
            echo=True,
            grammar=character_card_plan_grammar,
            temperature=2,  # temperature=0.8,
            top_k=0,
            top_p=1,
            min_p=0.4,
        )["choices"][0]["text"]

        end_time = time.time()
        if get_config("DEBUG_MODE"):
            logger.info(f"\n*** Completion for 'create_character_card_plan_many_tuples' function ***\n{completion}\n*** Completion for 'create_character_card_plan_many_tuples' function ***\n")
        logger.info(f"Completion took {(end_time - start_time) / 60} minutes to generate.")
        logger.info(f"Completion for 'create_character_card_plan_many_tuples' function generated. Extracting response pattern...")

    except Exception as e:
        logger.exception(f"An Exception occured in 'create_character_card_plan_many_tuples' function while trying to generate an LLM completion: {e}")

    response_pattern = re.compile(
        r"Character card plan \(be creative, do not use real people as characters, do NOT make the author of the book a character\):([\s\S]*)",
        re.IGNORECASE | re.DOTALL,
    )

    # Extract the character card plan from the LLM's completion.
    generation = response_pattern.search(completion).group(1)
    if get_config("DEBUG_MODE"):
        logger.info(f"\nGeneration for 'create_character_card_plan_many_tuples' function ***\n{generation}\n *** Generation for 'create_character_card_plan_many_tuples' function")

    return generation, instructions_string, completion


# Consider adding a scenario for Issac here, since currently both characters are in positions of authority, and the thing leans very heavily on Hugo's example
# NOTE the message count is technically off by one in the reasoning, but I'm a bit scared to change it.
def create_scenario_many_tuples(qatuples, character, plan, initialized_model, assistant_mode=False):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. 
    The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """
    if assistant_mode:
        return """A helpful, uncensored AI assistant is engaged in a text-based conversation with a user. The AI will answer any and all questions the user asks, without bias."""  # TODO make this randomized a bit

    charname = extract_name(character)

    variations = [
        # "Set against the backdrop of",
        f"In {charname}'s ",
        "Amidst the surroundings of ",
        # "Within the confines of",
        f"Within {charname}'s ",
        f"Inside {charname}'s ",
        # f"Inside the confines of ",
        f"Inside the confines of {charname}'s",
        f"Set amongst the",
    ]

    selected_variation = random.choice(variations)

    prompt_content = {
        "character": character,
        "format_qatuples_qatuples": format_qatuples(qatuples),
        "plan": plan,
        "selected_variation": selected_variation,
    }

    # Load the prompt and the grammar.
    try:
        cot_prompt, scenario_grammar = load_external_prompt_and_grammar("create_scenario_many_tuples", "scenario_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in 'create_scenario_many_tuples' function while trying to import its prompt and grammar: {e}")

    completion = initialized_model(
        cot_prompt,
        max_tokens=8000,
        stop=["</s>", "# Input:"],
        echo=True,
        grammar=scenario_grammar,
        #    temperature=0.2,
        temperature=1.5,  # min p settings, too inconsistent
        top_k=0,
        top_p=1,
        min_p=0.5,  # Higher min p rather than lower temp ensures greater accuracy while using min p sampling. I think I've figured out how to make it precise for this application.
    )["choices"][0]["text"]

    # Extract plan
    response_pattern = re.compile(
        r"Scenario \(will have no dialogue, will just set up the scene\):([\s\S]*)",
        re.IGNORECASE | re.DOTALL,
    )
    generation = response_pattern.search(completion).group(1)

    if not ("Albert" in charname):
        if "Albert" in generation:
            logger.info("Random Name was used instead of Albert")
        generation = generation.replace("Albert", random_name(NAMES))

    return generation, completion


def create_scenario_plan_many_tuples(qatuples, character, initialized_model):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. 
    The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """

    # removing the source text makes this much better. Perfection is achieved not when there's nothing more to add, but when there's nothing left to take away.

    charname = extract_name(character)
    prompt_content = {
        "format_qatuples_qatuples": format_qatuples(qatuples),
        "character": character
    }

    # Load the prompt and the grammar.
    try:
        cot_prompt, scenario_plan_many_tuples_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "scenario_plan_many_tuples_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

    # Even if the example does a justified clever trick, the model imitating it may fuck up the trick. 
    # So try to avoid complex things that aren't needed for the task in examples, like the "just how much have you dug" colloquialization
    completion = initialized_model(
        cot_prompt,
        max_tokens=8000,
        stop=["</s>", "# Input:"],
        echo=True,
        grammar=scenario_plan_many_tuples_grammar,
        temperature=1.5,  # min p settings, too inconsistent
        top_k=0,
        top_p=1,
        min_p=0.5,
    )["choices"][0]["text"]

    #logger.info(f"\n create_scenario_plan_many_tuples COMPLETION:\n\n-------------------\n\n {completion}\ncreate_scenario_plan_many_tuples COMPLETION:\n------------------)")
    #time.sleep(5)
    # Extract plan
    response_pattern = re.compile(
        r"Scenario plan \(be creative, and make sure all characters present fit in with the setting\):([\s\S]*)",
        re.IGNORECASE | re.DOTALL,
    )

    generation = response_pattern.search(completion).group(1)
    if get_config("DEBUG_MODE"):
        logger.info(f"\n create_scenario_plan_many_tuples GENERATION:\n\n-------------------\n\n {generation}\ncreate_scenario_plan_many_tuples GENERATION\n------------------)")
    time.sleep(5)

    # Cut-off pattern to find "Step 6: " and everything after it
    # This prevents ValueErrors, as sometimes the LLM blathers on and we need to truncate the completion to save on token space.
    cut_off_pattern = re.compile(
        r"Step 6:.*$", 
        re.DOTALL | re.IGNORECASE
    )

    # Check if the LLM blathered. If it did, truncate the completion.
    if cut_off_pattern.search(generation):
        generation = cut_off_pattern.sub('', generation)
        logger.warning("Warning: LLM blathered on. Truncating response to prevent context overflow errors...")
        print("You may want to check the multi-turn conversation info afterwards, as truncation may affect downstream LLM completions.")
        logger.info(f"\n create_scenario_plan_many_tuples BLATHERING GENERATION:\n\n-------------------\n\n {generation}\ncreate_scenario_plan_many_tuples BLATHERING GENERATION\n------------------)")
        time.sleep(3)


    # Change name from default "Albert".
    if not ("Albert" in charname):
        if "Albert" in generation:
            logger.info("Random Name was used instead of Albert")
        generation = generation.replace("Albert", random_name(NAMES))

    return generation.strip(), completion





# multiturn helpers
# These will probably be used for multiturn rapid-fire answering.

# Idea: use multiple short answers to train the task of answering multiple questions in one response. 
# Two-three short answers per response should be enough.
def make_multiturn_character(qa_tuples, conv_id, initialized_model):
    # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file
    if get_config("ASSISTANT_MODE"): 
        return "will_be_replaced", "will_be_replaced"

    # I will reuse the many tuples function for short question-answers, there's a lot of prompting in here already
    plan, instructions, card_plan_output = create_character_card_plan_many_tuples(qa_tuples, initialized_model)  
    write_output_to_file(card_plan_output, "./multiturn_card_plan_generations", conv_id)

    char, char_output = create_character_card_many_tuples(qa_tuples, plan, instructions, initialized_model)  # creates a character card
    write_output_to_file(char_output, "./multiturn_card_generations", conv_id)

    return char, instructions


def make_multiturn_conversation_info(qa_tuples, initialized_model):
    conv_id = make_id()

    # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file
    if get_config("ASSISTANT_MODE"): 
        return (qa_tuples, "will", "be", "replaced", conv_id)

    # thought_plan = create_thought_plan_many_tuples(qa_tuples,character,scenario,initialized_model)
    # There IS a way to make multiturn chain of thought answering work: 
    # Namely, generate each pair of messages using a separate prompt or a separate function, each of which has only the thought plan for that question/answer pair. 
    # But simply cramming in all the step-by-step things will confuse the hell out of the poor model. 
    # So for the first release version we're skipping it and just giving the response, with no reasoning, in the multiturn convs.
    character, instructions = make_multiturn_character(qa_tuples, conv_id, initialized_model)
    scenario, scenario_plan = make_multiturn_scenario(qa_tuples, character, conv_id, initialized_model )


def make_multiturn_scenario(qa_tuples, character, conv_id, initialized_model):
    max_retries = 3
    attempts = 0

    # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file.
    if get_config("ASSISTANT_MODE"):  
        return "will_be_replaced", "will_be_replaced"

    # Create a scenario plan based on a character card and a QA tuple.
    plan, scenario_plan_output = create_scenario_plan_many_tuples(qa_tuples, character, initialized_model)
    write_output_to_file(scenario_plan_output, "./multiturn_scenario_plan_generations", conv_id)

    # Create a scenario based on a character card, a scenario plan, and a QA tuple.
    scenario, scenario_output = create_scenario_many_tuples(qa_tuples, character, plan, initialized_model)  
    write_output_to_file(scenario_output, "./multiturn_scenario_generations", conv_id)

    return scenario, plan


class ReturnMultiturnConversationInfoSimple: # TODO Write function documentation. Oh, and finish the function. And I don't care what anyone else says, this WILL become multiple nodes so help me god!
    """
    This function 
    :param qa_tuples_by_paragraph: 
    :param multi_turn_convs_info_dir: 
    :param arrangements_to_take: How many of the possible permutations of tuples in a group to take and make multiturn conversations out of.
    :return multi_turn_convs_info:
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "qa_tuples_by_paragraph": ("TUPLE", {"forceInput": True}),
                "LLM": ("LLM",),
                "multi_turn_convs_info_dir": ("STRING", {"default": 'multi_turn_convs_info_dir'},),
                "assistant_mode_arg": (["Off","On"],),
                "rearrangements_to_take_arg": ("INT", {"default:": 3, "min": 1, "max": 10, "step":1},),
                "use_filenames_arg": (["False","True"],),
                "purge_loaded_llm_from_memory_after_node_is_done": (["Off","On"],),
                "use_config_file_presets_instead": (["True","False"],), 
            },
        }
    RETURN_TYPES = ("OUTPUT_TEXT","PURGE_TRIGGER",)
    RETURN_NAMES = ("multi_turn_convs_info",)
    FUNCTION = "return_multi_turn_conversation_info"

    CATEGORY = "augmentoolkit_functions"

    def return_multi_turn_conversation_info(self, 
                                            qa_tuples_by_paragraph, 
                                            LLM, 
                                            multi_turn_convs_info_dir, 
                                            assistant_mode_arg, 
                                            arrangements_to_take_arg,
                                            use_filenames_arg, 
                                            purge_loaded_llm_from_memory_after_node_is_done,
                                            use_config_file_presets_instead):

        # Determine where these variables' values come: the config.yml file, or the node itself.
        if use_config_file_presets_instead == "True": # Use presets from the config.yml file.
            assistant_mode = get_config("ASSISTANT_MODE")
            rearrangements_to_take = get_config("REARRANGEMENTS_TO_TAKE")
            use_filenames = get_config("USE_FILENAMES")
        else: # Use presets from the node.
            arrangements_to_take = arrangements_to_take_arg
            if assistant_mode_arg == "On":
                assistant_mode = True
            else:
                assistant_mode = False
            if use_filenames_arg == "True":
                use_filenames = True
            else:
                use_filenames = False
                
        try:
            if not os.path.exists(f"./{multi_turn_convs_info_dir}"):
                os.makedirs(f"./{multi_turn_convs_info_dir}")
        except Exception as e:
            logger.exception(f"ERROR: 'return_multi_turn_conversation_info' function could not create its directory using the name '{multi_turn_convs_info_dir}' due to: {e}. \n This was likely caused by invalid characters or syntax in the given folder name.")
            raise e

        multi_turn_convs_info = []

        if get_config("DEBUG_MODE"):
            logger.info(f"\n*** INPUT FOR qa_tuples_by_paragraph *** \n{qa_tuples_by_paragraph} \n*** INPUT FOR qa_tuples_by_paragraph *** ")

        if LLM['type'] == 'llamacpp':
            for idx, group in enumerate(qa_tuples_by_paragraph):
                logger.info(f"\n*** Current qa_tuples_by_paragraph *** \ngroup:{group} \nidx:{idx} ")

                all_permutations = list(itertools.permutations(group))
                sample_size = min(arrangements_to_take, len(all_permutations))
                sampled_permutations = random.sample(all_permutations, sample_size)

                if get_config("DEBUG_MODE"):
                    logger.info(f"all_permutations:\n{all_permutations}\nall_permutations")
                    logger.info(f"sample_size:\n{sample_size}\sample_size")
                    logger.info(f"sampled_permutations:\n{sampled_permutations}\sampled_permutations")

                group_convs_info = []

                for iter, perm in enumerate(sampled_permutations):
                    file_path = os.path.join(multi_turn_convs_info_dir, f"info_{idx}_{iter}.json")

                    # Skip if file already exists
                    if not os.path.exists(file_path):
                        print(f"Writing information for idx:{idx} to '{file_path}' in '{multi_turn_convs_info_dir}' directory...")
                        info = make_multiturn_conversation_info(perm, LLM)

                        if info is not None:
                            with open(file_path, "w") as file:
                                json.dump(info, file, indent=4)

                        group_convs_info.append(info)
                    else:
                        print(f"Skipped generating {file_path} as it already exists.")

                multi_turn_convs_info.append(group_convs_info)

        elif LLM['type'] == 'aphrodite':
            tasks = [
                aug_async.create_info(
                    idx,group,
                    LLM, 
                    assistant_mode, 
                    multi_turn_convs_info,
                    multi_turn_convs_info_dir, 
                    rearrangements_to_take=rearrangements_to_take,
                    use_filenames=use_filenames,  
                    logging_level=get_config("LOG_LEVEL")) for idx,group in enumerate(qa_tuples_by_paragraph)
            ]
            limited_tasks_infocreation = [run_task_with_limit(task) for task in tasks]
            asyncio.run(run_tasks(limited_tasks_infocreation))
                
        elif LLM['type'] == 'api':
            tasks = [
                aug_async.create_info(
                    idx,group,
                    LLM, 
                    assistant_mode, 
                    multi_turn_convs_info,
                    multi_turn_convs_info_dir, 
                    rearrangements_to_take=rearrangements_to_take,
                    use_filenames=use_filenames, 
                    logging_level=get_config("LOG_LEVEL")) for idx,group in enumerate(qa_tuples_by_paragraph)
            ]
            limited_tasks_infocreation = [run_task_with_limit(task) for task in tasks]
            asyncio.run(run_tasks(limited_tasks_infocreation))
        else:
            logger.error("ERROR: Unsupported LLM format passed to 'return_multi_turn_conversation_info' function in class ReturnMultiturnConversationInfoSimple.")

        # Set-off the purge trigger. Note that this does NOT automatically purge the loaded LLM from RAM and/or VRAM. 
        # Rather, it produces an boolean that tells the PurgeLlmFromRamOrVram node whether to do that or not.
        if purge_loaded_llm_from_memory_after_node_is_done == "On":
            purge_trigger = True
        else:
            purge_trigger = False

        return (multi_turn_convs_info, purge_trigger,)


#######################################


class GroupVettedTuplesByText:
    """
    This function groups QA tuples by paragraph.

    :param vetted_qa_tuples: Output from the ReviseQATuples function.
    :param check_for_matching_subtrings_anywhere: Check for matching substrings anywhere in the QA tuple.
    :return qa_tuples_by_paragraph: QA tuples grouped by paragraph.
    """

    @staticmethod
    # Group tuples for multiturn example generation (by chunk of source text) and then run that helper (so that we can make multiturn conversations from questions based on the same paragraphs)
    def group_by_text(tuples_list, check_for_matching_subtrings_anywhere=False):
        # Dictionary to hold the groups with text as the key
        groups = {}

        # Iterate over each tuple in the list
        for question, answer, text, textname in tuples_list:
            # If the text is not yet a key in the dictionary, add it with an empty list
            if text not in groups:
                groups[text] = []

            # Append the current tuple to the appropriate list
            groups[text].append((question, answer, text, textname))

        # Return the values of the dictionary, which are the lists of tuples grouped by text; also remove duplicates
        return [GroupVettedTuplesByText.identify_duplicates(group, check_for_matching_subtrings_anywhere) for group in list(groups.values())]

    @staticmethod
    def identify_duplicates(tuples: List[Tuple[str, str, str, str]], check_for_matching_subtrings_anywhere=False) -> List[Tuple[str, str, str, str]]:
        # If you want to check for matching substrings anywhere, not just at start, use this code (untested)
        if check_for_matching_subtrings_anywhere:
            # Create a dictionary to hold questions with the same first N characters
            question_dict = {}

            # Iterate through each tuple and categorize them by the first N characters of the question
            for q_tuple in tuples:
                question = q_tuple[0]
                placed = False

                for dict_q in question_dict.keys():
                    if has_sequential_chars(question,dict_q,get_config("N_CHARACTERS_SAME")):
                        question_dict[dict_q].append(q_tuple)
                        placed = True
                        break
                if not placed:
                    question_dict[question] = [q_tuple] # if not found to be equivalent with anything, make it a dict entry so that things can be compared against it and added to its list

            # Filter out prefixes that only have one question associated
            matching_questions = [q for q_list in question_dict.values() if len(q_list) > 1 for q in q_list]

            return matching_questions

        else:
            # Create a dictionary to hold questions with the same first N characters
            question_dict = {}

            # Iterate through each tuple and categorize them by the first N characters of the question
            for q_tuple in tuples:
                question = q_tuple[0]

                # Get the first N characters of the question
                prefix = question[:15]

                # Add the tuple to the list of tuples with the same prefix
                if prefix in question_dict:
                    question_dict[prefix].append(q_tuple)
                else:
                    question_dict[prefix] = [q_tuple]

            matching_questions = [q for q_list in question_dict.values() if len(q_list) == 1 for q in q_list]

            selected_from_duplicates = [q_list[0] for q_list in question_dict.values() if len(q_list) > 1]

            return matching_questions + selected_from_duplicates

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "vetted_qa_tuples": ("TUPLE", {"forceInput": True}),
                "check_for_matching_subtrings_anywhere": (["False","True"],),
            },
        }
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("qa_tuples_by_paragraph",)

    FUNCTION = "group_vetted_qa_tuple_by_paragraph"

    CATEGORY = "augmentoolkit_functions"

    def group_vetted_qa_tuple_by_paragraph(self, vetted_qa_tuples, check_for_matching_subtrings_anywhere):

        if check_for_matching_subtrings_anywhere == "True":
            qa_tuples_by_paragraph = self.group_by_text(vetted_qa_tuples, check_for_matching_subtrings_anywhere=True)
        else:
            qa_tuples_by_paragraph = self.group_by_text(vetted_qa_tuples)

        return(qa_tuples_by_paragraph,)


# TODO Setup the config so that it can pull the LLMs from outside the program i.e. if they're in a folder in Oobabooga or something.
class LlmLoaderAdvanced:
    """
    Load a specified llama-cpp model.

    :param model_name: The name of a llama-cpp model e.g. gguf.
    :param purge_trigger: A boolean that signals to the node that the specified llama-cpp model should be loaded.
    :return LLM: An LLM dictionary object, containing the loaded LLM and override and other meta-data.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("llm"),), # String
                "n_gqa_arg": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1},),
                "offload_kqv_arg": (["True", "False"],), # list
                "n_ctx_arg": ("INT", {"default": 12000, "min": 256, "max": 1000000, "step": 1},), # ONE MILLION CONTEXT!!!
                "rope_freq_scale_arg": ("FLOAT", {"default": 0.33, "min": 0.00, "max": 1.00, "step": 0.01},),
                "n_gpu_layers_arg": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1},),
                "verbose_arg": (["False", "True"],), # list
            },
            "optional": {
                "purge_trigger": ("PURGE_TRIGGER",), # Boolean
            },
        }
    RETURN_TYPES = ("LLM",)
    FUNCTION = "load_model"

    CATEGORY = "augmentoolkit_functions/loaders/advanced"

    @conditional_log_arguments
    def load_model(self, model_name, n_gqa_arg, offload_kqv_arg, n_ctx_arg, rope_freq_scale_arg, n_gpu_layers_arg, verbose_arg, purge_trigger=False):

        # Unload the model from RAM and VRAM if the purge trigger is true.
        # Note that since the purge is done in another node, 
        # the purge_trigger class input is solely to keep the function in the node chain.
        if purge_trigger:
            logger.info(f"Previous model cleared from RAM and/or VRAM. Loading {model_name}...")

        # Create an empty config_list.
        config_list = []

        logger.info(f"Loading Llama-cpp model'{model_name}'...")
        try: # Load the model with the specified parameters.
            llm = Llama(
                model_path=folder_paths.get_full_path("llm", model_name),
                n_gqa=n_gqa_arg,
                n_gpu_layers=n_gpu_layers_arg,
                offload_kqv=True if offload_kqv_arg == "True" else False,
                n_ctx=n_ctx_arg,
                rope_freq_scale=rope_freq_scale_arg,
                verbose=True if verbose_arg == "True" else False,
            )

            # Save the loaded model and base settings to config_list.
            config_list = [
                {
                'llm': llm,
                'type': 'llamacpp',
                'prompt': None,
                'grammar': None,
                'override_llm_presets': False
                }
            ]

            # Set the loaded_llm_name global variable
            folder_paths.set_loaded_llm_name(model_name)
            logger.info(f"global variable loaded_llm_name set to: {folder_paths.get_loaded_llm_name()}")
            time.sleep(3)

            # Record the model name and presets to the debug log.
            logger.info(f"Llama-cpp Model '{model_name}' succesfully loaded.")
            logger.info(f"\nLlama-cpp Model presets:\noffload_kqv={offload_kqv_arg}\nn_ctx={n_ctx_arg}\nrope_freq_scale={rope_freq_scale_arg}\nn_gpu_layers={n_gpu_layers_arg}\nverbose={verbose_arg}\n")
            time.sleep(3)

        except Exception as e:
            logger.exception(f"An Exception occured in load_model function in class LlmLoaderSimple: {e}")
            # We want to raise an exception here, since NOTHING can be done in augmentoolkit without LLM generations.
            raise e

        # Export the "LLM" dictionary object.
        return ({
            'llm': llm,
            'type': 'llamacpp',
            'prompt': None,
            'grammar': None,
            'override_llm_presets': False
        },)


# TODO Figure out what types the quantization argument can take without it breaking.
# TODO Create an OverrideNodePresetsAphrodite class.
class LlmLoaderAphroditeSimple:
    """
    This node loads in an aphrodite-engine model and exports it and its settings to connected downstream nodes.
    
    :param model_name:
    :param quantization:
    :return LLM:
    
    Requirements:
    from aphrodite import (
    EngineArgs,
    AphroditeEngine,
    SamplingParams,
    AsyncAphrodite,
    AsyncEngineArgs,
    )
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("STRING", {"default": 'FlatOrcamaid-13B-v0.2-GPTQ'},), # String
                "quantization": (["gptq","awq","gguf","quip","squeezellm"],), # List
            },
            "optional": {
            },
        }
    # ComfyUI will effectively return the Aphrodite class instantiation provided by engine_wrapper_aphrodite() and call it an LLM
    RETURN_TYPES = ("LLM",)
    FUNCTION = "engine_wrapper_aphrodite"

    CATEGORY = "augmentoolkit_functions/loaders"

    @conditional_log_arguments
    def engine_wrapper_aphrodite(self, model_name, quantization):

        # Set the loaded_llm_name global variable
        folder_paths.set_loaded_llm_name(model_name)
        logger.info(f"global variable loaded_llm_name set to: {folder_paths.get_loaded_llm_name()}")
        time.sleep(3)

        llm_dir = folder_paths.get_llm_directory()
        model_name = os.path.join(llm_dir, model_name)
        
        try: # Define the engine wrapper.
            engine_wrapper = EngineWrapper(model_name, mode="aphrodite", quantization=quantization)
        except Exception as e:
            logger.exception(f"An Exception occured in 'engine_wrapper_aphrodite' function in class LlmLoaderAphroditeSimple: {e}")
            raise e
        
        # Record the model name and presets to the debug log.
        logger.info(f"Aphrodite-Engine: Model {model_name} succesfully loaded.")
        logger.info(f"\nAphrodite-Engine Model presets:\nquantization={quantization}\nengine_use_ray=False\ndisable_log_requests=True\nmax_model_len=12000\ndtype='float16'\n")
        time.sleep(3)
        
        # Put the engine wrapper and other arguments into the config_list
        config_list = [
            {
            'llm': engine_wrapper,
            'type': 'aphrodite',
            'prompt': None,
            'sampling_params': None,
            'override_aphrodite_sampling_presets': False,
            }
        ]

        # Export the "LLM" dictionary object.
        return ({
            'llm': engine_wrapper,
            'type': 'aphrodite',
            'prompt': None,
            'sampling_params': None,
            'override_aphrodite_sampling_presets': False,
        },)


class LlmLoaderSimple:
    """
    Load a specified llama-cpp model.
    This node's settings are hardcoded to match augmentoolkit's original settings.
    :param model_name: The name of a llama-cpp model e.g. gguf.
    :param purge_trigger: A boolean that signals to the node that the specified llama-cpp model should be loaded.
    :return LLM: An LLM dictionary object, containing the loaded LLM and override meta-data.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("llm"),), # String
            },
            "optional": {
                "purge_trigger": ("PURGE_TRIGGER",), # Boolean
            },
        }
    RETURN_TYPES = ("LLM",)
    FUNCTION = "load_model"

    CATEGORY = "augmentoolkit_functions/loaders"

    @conditional_log_arguments
    def load_model(self, model_name, purge_trigger=False):

        # Unload the model from RAM and VRAM if the purge trigger is true.
        # Note that since the purge is done in another node, 
        # the purge_trigger class input is a dummy input that allows this node to act as both a beginning and intermediate linkage in a workflow.
        if purge_trigger:
            logger.info(f"Previous model cleared from RAM and/or VRAM. Loading {model_name}...")

        # Create an empty config_list.
        config_list = []

        logger.info(f"Loading Model {model_name}...")
        try:
            # Load the LLM, then put it in the config_list.
            llm = Llama(
                model_path=folder_paths.get_full_path("llm", model_name),
                offload_kqv=True,
                n_ctx=12000,
                rope_freq_scale=0.33,
                n_gpu_layers=100,
                verbose=False,
            )

            config_list = [
                {
                'llm': llm,
                'type': 'llamacpp',
                'prompt': None,
                'grammar': None,
                'override_llm_presets': False
                }
            ]

            # Set the loaded_llm_name global variable
            folder_paths.set_loaded_llm_name(model_name)
            logger.info(f"global variable loaded_llm_name set to: {folder_paths.get_loaded_llm_name()}")
            time.sleep(3)

            # Record the model name and presets to the debug log.
            logger.info(f"Model {model_name} succesfully loaded.")
            logger.info(f"\nModel presets:\noffload=True\nn_ctx=12000\nrope_freq_scale=0.33\nn_gpu_layers=100\nverbose=False\n")
            time.sleep(3)

        except Exception as e:
            logger.exception(f"An Exception occured in load_model function in class LlmLoaderSimple: {e}")
            # We want to raise an exception here, since NOTHING can be done in augmentoolkit without LLM generations.
            raise e

        # Export the "LLM" dictionary object.
        return ({
            'llm': llm,
            'type': 'llamacpp',
            'prompt': None,
            'grammar': None,
            'override_llm_presets': False
        },)


#TODO Make sure the preset defaults are proper.
# Write function documentation.
class OverrideLlmPresetsInConnectedNodeAphrodite:
    """ 
    This node overrides the LLM presets in its downstream node with the given parameter settings.
    Aphrodite needs its own override node because of different settings arguments between llama-cpp and aphrodite-engine.
    Options also exist to override the downstream node's prompt.
    Useful for experimentation or debugging prompts and LLM presets.
    All settings are set to "off" by default.
    
    :param model: The loaded LLM (external)
    :param max_tokens: The maximum number of tokens that can be generated in a single run.
    :param max_token_length: The maximum token length for a chunk of sentences
    :param stop_arg: The specified stop tokens, usually ["</s>", "# Input:"] 
    :param echo_arg: 
    :param temperature_arg: The specified temperature. Scales token probabilities so that the 2nd, 3rd, etc most-likely tokens have a greater likelihood of being selected.
    :param top_k_arg: 
    :param top_p_arg: 
    :param min_p_arg: 
    :param seed_arg: 
    :return llm: an initialized model.
    :return prompt: a prompt to go with the initialized model
    :return sampling_params: sampling parameters to go with the initialized model
    :return override_aphrodite_sampling_presets: a boolean signaling whether or not the nodes presets need to be overruled.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LLM": ("LLM",),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.00, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.00, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "top_k": ("INT", {"default": 40, "min":1, "max":100, "step":1}),
                "top_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "min_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "tfs": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "eta_cutoff": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "epsilon_cutoff": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "mirostat_mode": ("INT", {"default": 40, "min":1, "max":100, "step":1}),
                "mirostat_tau": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "mirostat_eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "dynatemp_range": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "dynatemp_exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "smoothing_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "use_beam_search": (['False'],['True']),
                "length_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "early_stopping": ("STRING", {"default": '["</s>", "# Input:"]'}),
                "stop": ("STRING", {"default": '["</s>", "# Input:"]'}),
                "stop_token_ids": ("STRING", {"default": '["</s>", "# Input:"]'}),
                "include_stop_str_in_output": (['False'],['True']),
                "ignore_eos": (['False'],['True']),
                "max_tokens": ("INT", {"default": 4000, "min":1, "max":1000000, "step":1}),
                "logprobs": ("INT", {"default": 40, "min":1, "max":100, "step":1}),
                "prompt_logprobs": ("INT", {"default": 40, "min":1, "max":100, "step":1}),
                "custom_token_bans": ("STRING", {"default": '["1", "2"]'}),
                "skip_special_tokens": (['True'],['False']),
                "spaces_between_special_tokens": (['True'],['False']),
                "logits_processors": ("STRING", {"default": '["</s>", "# Input:"]'}),
                "only_override_prompt": (["False", "True"],), # This allows the prompt to be overriden while still keeping the original function generation presets.
            },
            "optional": {
                "prompt": ("PROMPT", {"forceInput": True}),
            },
        }
    RETURN_TYPES = ("LLM",)
    FUNCTION = "override_llm_presets_in_connected_node"

    CATEGORY = "augmentoolkit_functions/advanced/overrides"

    @conditional_log_arguments
    def override_llm_presets_in_connected_node(self, LLM, presence_penalty, frequency_penalty,
                                               repetition_penalty, temperature, top_p, top_k,
                                               top_a, min_p, tfs, eta_cutoff, epsilon_cutoff, typical_p,
                                               mirostat_mode, mirostat_tau, mirostat_eta, dynatemp_range, 
                                               dynatemp_exponent, smoothing_factor, use_beam_search, length_penalty,
                                               early_stopping, stop, stop_token_ids, include_stop_str_in_output,
                                               ignore_eos, max_tokens, logprobs, prompt_logprobs, custom_token_bans,
                                               skip_special_tokens, spaces_between_special_tokens, logits_processors,
                                               only_override_prompt, prompt=None):
        # Create an empty config_list.
        config_list = []

        # Extract the loaded model from the LLM dictionary.
        # This corresponds to 'engine_wrapper' in the upstream node.
        model = LLM['llm']

        if only_override_prompt == "True" and prompt is None:
            logger.warning("Warning: Current node settings will generate an empty LLM object. Defaulting to downstream node presets.")
            config_list = [ # Pass through the original model.
                    {
                        'llm': model,
                        'type': 'aphrodite',
                        'prompt': None,
                        'sampling_params': None,
                        'override_aphrodite_sampling_presets': False
                    }
                ]

        try:
            # Create a partial function and call it sampling_params. 
            # Note that this is for aphrodite-engine
            sampling_params = partial(SamplingParams,  # First argument of partial is the function we want to fix some of the arguments for.
                          presence_penalty=presence_penalty, 
                          frequency_penalty=frequency_penalty, 
                          repetition_penalty=repetition_penalty, 
                          temperature=temperature, 
                          top_p=top_p, 
                          top_k=top_k, 
                          top_a=top_a, 
                          min_p=min_p, 
                          tfs=tfs, 
                          eta_cutoff=eta_cutoff, 
                          epsilon_cutoff=epsilon_cutoff, 
                          typical_p=typical_p, 
                          mirostat_mode=mirostat_mode, 
                          mirostat_tau=mirostat_tau, 
                          mirostat_eta=mirostat_eta, 
                          dynatemp_range=dynatemp_range, 
                          dynatemp_exponent=dynatemp_exponent, 
                          smoothing_factor=smoothing_factor, 
                          use_beam_search=False if use_beam_search == "False" else True, 
                          length_penalty=length_penalty,
                          early_stopping=early_stopping, 
                          stop=stop, 
                          stop_token_ids=stop_token_ids, 
                          include_stop_str_in_output=False if include_stop_str_in_output == "False" else True,
                          ignore_eos=False if ignore_eos == "False" else True, 
                          max_tokens=max_tokens, 
                          logprobs=logprobs, 
                          prompt_logprobs=prompt_logprobs, 
                          custom_token_bans=custom_token_bans,
                          skip_special_tokens=skip_special_tokens, 
                          spaces_between_special_tokens=spaces_between_special_tokens, 
                          logits_processors=logits_processors,
            )
            logger.info(f"Aphrodite LLM override parameters set.")
            if get_config("DEBUG_MODE"): #TODO Set this debug up
                logger.info("Current Aphrodite-Engine Sampling Parameters:")
                for key, value in sampling_params.keywords.items():
                    logger.info(f"    {key}: {value}")
            time.sleep(3)

            # Set up a config list to be exported under the "LLM" object type.
            # This creates a dictionary whose contents are the initialized model and its overrides.
            config_list = [
                {
                    'llm': model,
                    'type': 'aphrodite',
                    'prompt': None if prompt is None else prompt,
                    'sampling_params': None if only_override_prompt == "True" else sampling_params,
                    'override_aphrodite_sampling_presets': True if only_override_prompt == "False" else False
                }
            ]

            # Debug messages that record the config_list contents.
            if prompt is None:
                logger.info(f"LLM parameter override settings: \nLLM Settings: {sampling_params}\nPrompt: NA\n")
            elif prompt is not None and only_override_prompt == "True":
                logger.info(f"LLM parameter override settings: \nLLM Settings: NA\nPrompt: {prompt}\n")
            else:
                logger.info(f"LLM parameter override settings: \nLLM Settings: {sampling_params}\nPrompt: {prompt}\n")

        except Exception as e:
            logger.exception(f"An Exception occurred in 'override_llm_presets_in_connected_node' function in class OverrideLlmPresetsInConnectedNodeAphrodite: {e}")

        # Export the "LLM" dictionary object.
        return ({config_list},)


#TODO This might cause problems if the nodes its hooked up to are run simultaneously, as it will be calling the same model with two different inputs at the same time.
#TODO Set up overrides in nodes with INITIALIZED_MODEL inputs
#TODO Investigate the limitations of partial functions.
#TODO Finish documenting the function.
#TODO Set up other override nodes
class OverrideLlmPresetsInConnectedNodeLlama: 
    """ 
    This node overrides the LLM presets in its downstream node with the given parameter settings.
    Options also exist to override the downstream node's prompt and grammar.
    Useful for debugging prompts, grammars and LLM presets.
    All settings are set to "off" by default. Maybe change these to match oobas or something? - KR
    
    :param model: The loaded LLM (external)
    :param max_tokens: The maximum number of tokens that can be generated in a single run.
    :param max_token_length: The maximum token length for a chunk of sentences
    :param stop_arg: The specified stop tokens, usually ["</s>", "# Input:"] 
    :param echo_arg: 
    :param temperature_arg: The specified temperature. Scales token probabilities so that the 2nd, 3rd, etc most-likely tokens have a greater likelihood of being selected.
    :param top_k_arg: 
    :param top_p_arg: 
    :param min_p_arg: 
    :param seed_arg: 
    :return llm: an initialized model with the given parameter generation settings.
    :return prompt: a prompt to go with the initialized model
    :return grammar: a grammar to go with the initialized model
    :return override_llm_presets: a boolean signaling whether or not the nodes presets need to be overruled.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LLM": ("LLM",),
                "max_tokens_arg": ("INT", {"default": 6000, "min":1, "max":100000, "step":1}),
                "stop_arg": ("STRING", {"default": '["</s>", "# Input:"]'}),
                "echo_arg": (["True", "False"],),
                "temperature_arg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}), # Default is 0, which is deterministic e.g. the most likely token is always selected.
                "top_k_arg": ("INT", {"default": 40, "min":1, "max":1000, "step":1}), # Default is 40, which means only the top 40 token probabilities are considered for selection.
                "top_p_arg": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 20.0, "step": 0.01}), # Default is 1, which is off (???)
                "min_p_arg": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}), # Default is 0, which is off (???)
                "seed_arg": ("INT", {"default": -1, "min": -1, "max":0xffffffffffffffff, "step":1}), # Default is -1 i.e. the seed is randomly generated.
                "only_override_prompt_and_grammar": (["False", "True"],), # This allows the prompt and grammar to be overriden while still keeping the original function generation presets.
            },
            "optional": {
                "prompt": ("PROMPT", {"forceInput": True}),
                "grammar": ("GRAMMAR", {"forceInput": True}),
            },
        }
    RETURN_TYPES = ("LLM",)
    FUNCTION = "override_llm_presets_in_connected_node"

    CATEGORY = "augmentoolkit_functions/advanced/overrides"

    @conditional_log_arguments
    def override_llm_presets_in_connected_node(self, 
                                               LLM, max_tokens_arg, stop_arg, echo_arg, 
                                               temperature_arg, top_k_arg, top_p_arg, min_p_arg, seed_arg, 
                                               only_override_prompt_and_grammar, 
                                               prompt=None, 
                                               grammar=None):

        # Create an empty config_list.
        config_list = []

        # Extract the loaded model from the LLM dictionary.
        model = LLM['llm']

        if only_override_prompt_and_grammar == "False" and prompt is None and grammar is None:
            logger.warning("Current node settings will generate an empty LLM object. Defaulting to downstream node presets.")
            config_list = [ # Pass through the original model.
                    {
                        'llm': model,
                        'type': 'llamacpp',
                        'prompt': None,
                        'grammar': None,
                        'override_llm_presets': False
                    }
                ]
        else:
            try:
                # Create a partial function and call it llm. Note that this is for a llama-cpp model.
                llm = partial(model,  # First argument of partial is the function we want to fix some of the arguments for.
                    max_tokens=max_tokens_arg,
                    stop=stop_arg,
                    echo= True if echo_arg == "True" else False,
                    temperature=temperature_arg,
                    top_k=top_k_arg,
                    top_p=top_p_arg,
                    min_p=min_p_arg,
                    seed=seed_arg,
                )
                logger.info(f"Llama-cpp LLM override parameters set.")
                time.sleep(3)

                # Set up a config list to be exported under the "LLM" object type.
                # This creates a dictionary whose contents are the initialized model and overrides.
                config_list = [
                    {
                        'llm': llm,
                        'type': 'llamacpp',
                        'prompt': None if prompt is None else prompt,
                        'grammar': None if grammar is None else grammar,
                        'override_llm_presets': True if only_override_prompt_and_grammar == "False" else False
                    }
                ]   

            # Debug messages that record the config_list contents.
                if prompt is None and grammar is None:
                    logger.info(f"LLM parameter override settings: \nLLM Settings: {llm}\nPrompt: NA\nGrammar: NA")
                elif prompt is not None and grammar is None:
                    logger.info(f"LLM parameter override settings: \nLLM Settings: {llm}\nPrompt: {prompt}\nGrammar: NA")
                elif prompt is None and grammar is not None:
                    logger.info(f"LLM parameter override settings: \nLLM Settings: {llm}\nPrompt: NA\nGrammar: {grammar}")
                elif prompt is not None and grammar is not None and only_override_prompt_and_grammar == "True":
                    logger.info(f"LLM parameter override settings: \nLLM Settings: NA\nPrompt: {prompt}\nGrammar: {grammar}")
                else:
                    logger.info(f"LLM parameter override settings: \nLLM Settings: {llm}\nPrompt: {prompt}\nGrammar: {grammar}")

            except Exception as e:
                logger.exception(f"An Exception occurred in 'override_llm_presets_in_connected_node' function in class OverrideLlmPresetsInNode: {e}")

        # Export the "LLM" dictionary object.
        return ({"LLM": config_list},)


class PurgeLlmFromRamOrVram: 
    """
    This node purges the LLM from RAM or VRAM, since llama-cpp-python has a memory leak problem.
    Also, two-step generation.
    :param model: The loaded LLM  (external)
    :param purge_trigger: A boolean input that triggers the purge automatically.
    :return: NONE: The output is a clean RAM and/or VRAM cache!
    :return: reload_model: A boolean that goes to a loader node to signal that RAM or VRAM is cleared.
    """
    def __init__(self):
        self.loaded_model = folder_paths.get_loaded_llm_name()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "purge_trigger": ("PURGE_TRIGGER",),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "purge_llm_from_ram_or_vram"

    OUTPUT_NODE = True

    CATEGORY = "augmentoolkit_functions/advanced"

    def purge_llm_from_ram_or_vram(self, purge_trigger=False):
        reload_model = False
        if purge_trigger:
            # Always double-tap...
            release_memory(self.loaded_model)
            del self.loaded_model
            reload_model = True
        
        return (None, reload_model,)


class WriteOutputToFile: #TODO: Get this so that it can save multiple output texts in succession. 
    """
    This node outputs a text or json file to the output directory.

    :param output_text: the text to be output (external)
    :param output_tuple: the tuples to be output (external)
    :param filename_prefix: the prefix you want to stick onto the front of the output file.
    :return: NONE: the output file is the return for this function!
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "filename_prefix": ("STRING", {"default": "augmentoolkit"},),
                "file_output_type": (["txt","json"],),
            },
            "optional": {
                "output_text": ("STRING",),
                "output_tuple": ("TUPLE", {"forceInput": True}),
            },
        }
    RETURN_TYPES = ()
    FUNCTION = "write_output_to_file"

    OUTPUT_NODE = True

    CATEGORY = "augmentoolkit_functions"

    def write_output_to_file(self, filename_prefix, file_output_type, output_text=None, output_tuple=None) -> None:
        filename_prefix += self.prefix_append
        uuid = make_id()
        file_path = os.path.join(self.output_dir, f"{filename_prefix}_{uuid}.{file_output_type}")
        results = list()

        #print(output_text)
        # Write the output to the file
        try:
            if output_text is not None:
                with open(file_path, "w", encoding='utf-8-sig') as file:
                    for item in output_text:
                        if isinstance(item, list):
                            # If the item is a list, write each element on a new line
                            for sub_item in item:
                                file.write(str(sub_item) + "\n")
                        else:
                            # For non-list items, convert to string and write directly
                            file.write(str(item) + "\n")

            elif output_tuple is not None:
                with open(file_path, "w", encoding='utf-8-sig') as file:
                        json.dump(output_tuple, file)

            elif output_text is not None and output_text is not None:
                logger.error(f"ERROR: write_output_to_file function can only take a output_tuple or output_text as an argument, not both.")

            logger.info(f"Output written to {file_path}")

            results.append({
                "filename": f"{filename_prefix}_{uuid}",
                "folder": file_path,
                "type": self.type
            })
            logger.info(f"Results: {results}")

        except Exception as e:
            logger.exception(f"An Exception occured in write_output_to_file function when writing output to {file_path}: {e}")

        return { "ui": { "results": results } }


# TODO Write function documentation.
class InputTextLoaderSimple:
    """
    """
    def __init__(self):
        self.input_dir = folder_paths.get_input_directory()
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": { 
                "text_name": (sorted(files), {"text_upload": True}),
            },
        }
    RETURN_TYPES = ("RAW_TEXT",)
    FUNCTION = "load_text"

    CATEGORY = "augmentoolkit_functions/loaders"
    
    def load_text(self, text_name):
        text_path = os.path.join(self.input_dir, f"{text_name}")

        folder_paths.set_raw_text_name(text_name)
        logger.info(f"global variable raw_text_name set to: {folder_paths.get_raw_text_name()}")
        time.sleep(1)

        try:
            with open(text_path, "r", encoding="utf-8-sig") as file:
                raw_text = file.read()
                logger.info(f"Input text successfully loaded.")
                time.sleep(1)
            return (raw_text,) #So hacky...
        except Exception as e:
            logger.exception(f"An Exception occured in load_text function in InputTextLoaderSimple class: {e}")
            logger.info(f"Chosen text '{folder_paths.get_raw_text_name()}' failed to load!")


#######################################


class SelectRandomPrompt:
    """
    This node selects a random prompt from the inputs arguments. Includes a seed argument for reproducibility.
    Useful for conducting blinded studies of prompts.
    :param prompt1: A default prompt.
    :param prompt2: A second prompt.
    :param prompt3: A third prompt.
    :param prompt4: A fourth prompt.
    :return selected_prompt: The randomly chosen prompt.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "prompt1": ("PROMPT",), #String
                "prompt2": ("PROMPT",), #String
                "seed": ("INT", {"default": -1, "min": -1, "max":0xffffffffffffffff, "step":1}),
            },
            "optional": {
                "prompt3": ("PROMPT",), #String
                "prompt4": ("PROMPT",), #String
            },
        }
    RETURN_TYPES = ("PROMPT",)
    FUNCTION = "select_random_prompt"

    CATEGORY = "augmentoolkit_functions/debug"

    @log_arguments
    def select_random_prompt(self, prompt1, prompt2, seed, prompt3=None, prompt4=None):

        try:
            # Set a seed for reproducibility.
            if seed == "-1":
                random.seed()
            else:
                random.seed(seed)

            # Pair prompts with their variable names.
            prompts = {
                'prompt1': prompt1,
                'prompt2': prompt2
            }
            if prompt3 is not None:
                prompts['prompt3'] = prompt3
            if prompt4 is not None:
                prompts['prompt4'] = prompt4
        
            # Randomly select one prompt variable name from the list of keys.
            random_choice = random.choice(list(prompts.keys()))
        
            # Retrieve the selected prompt.
            selected_prompt = prompts[random_choice]
        
            # Retrieve which variable was chosen.
            random_choice = random.choice(prompts)
        
            logger.info(f"\n'{random_choice}' was selected with seed {seed} in 'select_random_prompt' function.\n *** PROMPT *** \n{random_choice}\n *** PROMPT ***\n")

            return (selected_prompt,)
        
        except Exception as e:
            logger.exception(f"An Exception occured in 'select_random_prompt' function in class 'SelectRandomPrompt': {e}")
            logger.info("Output will default to prompt1.")
            return (prompt1,)


#TODO Write the function.
class SelectRandomLlm:
    pass



###############################################
#### NODE CLASS MAPPINGS AND DISPLAY NAMES ####
###############################################

# This is intentional, as node classes cannot have the same name.
NODE_CLASS_MAPPINGS = {
    "ChunkParagraphs": ChunkParagraphs,
    "DisplayOutputText": DisplayOutputText,
    "FilterAndGraph": FilterAndGraph,
    "GenerateQATuplesSimple": GenerateQATuplesSimple,
    "GenerateQATuplesAdvanced": GenerateQATuplesAdvanced,
    "GroupVettedTuplesByText": GroupVettedTuplesByText,
    "InputTextLoaderSimple": InputTextLoaderSimple,
    "JudgeParagraphs": JudgeParagraphs,
    "LlmLoaderAphroditeSimple": LlmLoaderAphroditeSimple,
    "LlmLoaderSimple": LlmLoaderSimple,
    "LlmLoaderAdvanced": LlmLoaderAdvanced,
    "MakeDatasetMultiturnConversationSimple": MakeDatasetMultiturnConversationSimple,
    "OverrideLlmPresetsInConnectedNodeLlama": OverrideLlmPresetsInConnectedNodeLlama,
    "OverrideLlmPresetsInConnectedNodeAphrodite": OverrideLlmPresetsInConnectedNodeAphrodite,
    "ReturnMultiturnConversationInfoSimple": ReturnMultiturnConversationInfoSimple,
    "ReviseQATuples": ReviseQATuples,
    "WriteOutputToFile": WriteOutputToFile,
}
  #"OverrideLlmPresetsInConnectedNodeOpenai": OverrideLlmPresetsInConnectedNodeOpenai,
#     "ReturnMultiturnConversationInfoAdvanced": ReturnMultiturnConversationInfoAdvanced,


NODE_DISPLAY_NAME_MAPPINGS = {
    "ChunkParagraphs":"Chunk Plain-text File into Paragraphs",
    "DisplayOutputText": "Display Output Text in Console",
    "FilterAndGraph": "Filter and Graph Judged Paragraphs",
    "GenerateQATuplesSimple": "Generate QA Tuples (Simple)",
    "GenerateQATuplesAdvanced": "Generate QA Tuples (Advanced)",
    "GroupVettedTuplesByText": "Group Revised QA Tuples by Paragraph",
    "InputTextLoaderSimple": "Load Input Text",
    "JudgeParagraphs": "Judge Paragraphs for QA Suitability",
    "LlmLoaderAdvanced": "Load LLM (Advanced) (Llama-cpp-python)",
    "LlmLoaderAphroditeSimple": "Load LLM (Simple) (Aphrodite)",
    "LlmLoaderSimple": "Load LLM (Simple) (Llama-cpp-python)",
    "MakeDatasetMultiturnConversationSimple": "Make Dataset: Multi-turn Conversation",
    "OverrideLlmPresetsInConnectedNodeLlama": "Override LLM Presets in Connected Node(s) (Llama-cpp-python)",
    "OverrideLlmPresetsInConnectedNodeAphrodite": "Override LLM Prests in Connected Node(s) (Aphrodite)",
    #"OverrideLlmPresetsInConnectedNodeOpenai": "Override LLM Presets in Connected Node(s) (OpenAI)",
    "ReturnMultiturnConversationInfoAdvanced": "Create Multi-turn Conversation Info (Advanced)",
    "ReturnMultiturnConversationInfoSimple": "Create Multi-turn Conversation Info (Simple)",
    "ReviseQATuples": "Revise QA Tuples (Simple)",
    "WriteOutputToFile": "Write Output to File",
}


#############################
#### OTHER ODDS AND ENDS ####
#############################

