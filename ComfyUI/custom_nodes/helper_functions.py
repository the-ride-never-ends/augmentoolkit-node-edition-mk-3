import ast
import asyncio
import io
import json
import logging
import os
import re
import random
import sys
import time
import traceback
import uuid

from accelerate.utils import release_memory
from datetime import datetime
from functools import partial
from llama_cpp import Llama, LlamaGrammar
from typing import List, Tuple

from custom_nodes import grammars
from custom_nodes.logger import logger
import folder_paths
"""
from aphrodite import (
    EngineArgs,
    AphroditeEngine,
    SamplingParams,
    AsyncAphrodite,
    AsyncEngineArgs,
)
"""
# TODO: Organize all the functions from here and move them to their respective files. If not, add in the helper_nodes.function shit.

#########################################
#### PROMPT AND GRAMMAR DICTIONARIES ####
#########################################

# Create dictionaries for the prompts and grammars from the txt files in the prompts and grammars folders. 
# The prompt names from the dictionary must match the name of the function they go to.

PROMPT_DICT = {}
for file_name in folder_paths.get_filename_list("prompts"):
    try:
        file_path = folder_paths.get_full_path("prompts", file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            key = os.path.splitext(file_name)[0]
            PROMPT_DICT[key] = file.read()
    except Exception as e:
        logger.exception(f"An Exception occured when creating the prompt dictionary object: {e} ")

GRAMMAR_DICT = {}
for file_name in folder_paths.get_filename_list("grammars"):
    try:
        file_path = folder_paths.get_full_path("grammars", file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            key = os.path.splitext(file_name)[0]
            GRAMMAR_DICT[key] = file.read()
    except Exception as e:
        logger.exception(f"An Exception occured when creating the grammar dictionary object: {e} ")

"""
class EngineWrapper:
    def __init__(self, model, quantization):
        engine_args = AsyncEngineArgs(
            model=model,
            quantization=quantization,
            engine_use_ray=False,
            disable_log_requests=True,
            max_model_len=12000,
            dtype="float16"
        )
        self.engine = AsyncAphrodite.from_engine_args(engine_args)

    async def submit(self, prompt, sampling_params):  # Submit request and wait for it to stream back fully
        request_id = make_id()
        outputs = []
        # self.engine.add_request(request_id,prompt,sampling_params) #old sync code
        final_output = None
        async for request_output in self.engine.generate(
            prompt, sampling_params, request_id
        ):
            outputs.append(request_output.outputs[0].text)
            final_output = request_output

        full_output = "".join(outputs)
        return final_output.prompt + final_output.outputs[0].text
"""

##############################
#### INDIVIDUAL FUNCTIONS ####
##############################

def extract_name(str):
    # Regular expression to match 'Name:' followed by any characters until the end of the line
    name_regex = r"^Name:\s*(.*)$"

    # Searching in the multiline string
    match = re.search(name_regex, str, re.MULTILINE)

    if match:
        name = match.group(1)
        logger.info(f"Extracted name: {name}")
        return name
    else:
        logger.info("No name found")

# For the reword step (ONLY USE IF JUDGEMENT IS REWORD, OTHERWISE WE JUST IGNORE THE LAST BIT OF THE GEN)
def extract_question_answer(response):
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
        return None, None

# This function is dangerous too! But it is also critical to the running of the program.
# Otherwise, all the prompt f-strings have to be hard-wired into the code.
def format_external_text_like_f_string(external_text, prompt_content):
    pattern = r'{([a-zA-Z0-9_]+(\[[0-9]+\]){0,2}|\w+\(\))}'
    #pattern = r'{([a-zA-Z0-9_]+(\[[0-9]+\])?|\w+\(\))}'

    def replacer(match):
        placeholder = match.group(1)

        try:
            value = eval(placeholder, {}, prompt_content)
            return str(value)
        except (KeyError, IndexError, TypeError, SyntaxError, NameError) as e:
            return match.group(0)
            logger.exception(f"An Exception occured in format_external_text_like_f_string function using original placeholder {placeholder}: {e}")

    return re.sub(pattern, replacer, external_text)

# IMPORTANT FUNCTION: Do not change lightly as it's used by every node that relies on LLM generations.
def load_external_prompt_and_grammar(function_name, grammar_name, prompt_content: dict) -> Tuple[str, str]:

    # Format the function and grammar names as strings, if they're not ones already.
    grammar_name = str(grammar_name)
    function_name = str(function_name)

    print(f"Loading prompt and grammar for {function_name} function...")

    # Load the prompt and the grammar.
    try:
        prompt = format_external_text_like_f_string(PROMPT_DICT[f'{function_name}'], prompt_content) # Since we're importing the prompts from txt files, we can't use the regular f-string feature.
        print(f"Prompt for {function_name} loaded successfully.")
    except Exception as e:
        logger.exception(f"An Exception occurred in '{function_name}' function while trying to import its prompt: {e}")
        print(f"Check the prompt folder. The prompt must be a txt file named '{function_name}', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    try:
        # Redirect stdout and stderr to suppress output from LlamaGrammar.from_string()
        # Otherwise, it prints the entire grammar to the console and makes the debug logs a lot longer than they need to be.
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        grammar = LlamaGrammar.from_string(fr"{GRAMMAR_DICT[f'{grammar_name}']}")

        # Restore stdout and stderr after the grammar is loaded.
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Grammar for {function_name} loaded successfully.")

    except Exception as e:
        # Restore stdout and stderr in case of an exception before logging/handling the exception
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        logger.exception(f"An Exception occurred in '{function_name}' function while trying to import its grammar: {e}")
        print(f"Check the grammar folder. The grammar must be a txt file named '{grammar_name}'.")

    return prompt, grammar

def make_id():
    return str(uuid.uuid4())

# TODO Add the prompt and grammar back in.
def sanity_check(initialized_llm):
    retries = 0
    while retries <= 4:

        decision_prompt = f"""Hi there, """
        # print("DEBUG\n\n" + decision_prompt)
        completion = initialized_llm(
            decision_prompt,
            max_tokens=100,
            stop=["</s>", "# Input:"],
            echo=True,
            grammar=Grammars.answer_accurate_grammar,
            temperature=0.2,
        )["choices"][0]["text"]
        # print(completion)

        return

def write_output_to_file(output, directory, uuid):
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path using the directory and UUID
    file_path = os.path.join(directory, f"{uuid}.txt")

    # Write the output to the file
    with open(file_path, "w", encoding='utf-8') as file:
        file.write(output)

    print(f"Output written to {file_path}")

#########################################################
#### FUNCTIONS FROM ensure_multiple_answers_are_same ####
#########################################################

def has_sequential_chars(string1, string2, n):
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

def extract_conversation(conversation):
    """
    Extracts conversation from a string and returns it as a list of tuples.

    Parameters:
    conversation (str): A string representing the conversation.

    Returns:
    list of tuples: Each tuple contains the character's name and their message.
    """
    lines = conversation.strip().split("\n")
    dialogues = []

    for line in lines:
        if ":" in line:
            # Splitting at the first occurrence of ':'
            parts = line.split(":", 1)
            charname = parts[0].strip()
            message = parts[1].strip() if len(parts) > 1 else ""
            dialogues.append((charname, message))

    return dialogues

def compare_answers_with_qatuples(dialogues, qatuples, n):
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

def check_for_repeated_dialogue_answers(dialogues, qatuples, n):
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

"""
# def check_repeated_answer(dialogues, qatuples):
#     # Get the length of the dialogues
#     conv_length = len(dialogues)

#     # Loop through even indices starting from 2 (first answer is at index 2)
#     for i in range(2, conv_length, 2):
#         current_answer = dialogues[i][1][:n_characters_same]
#         next_answer_index = i + 2

#         if next_answer_index < conv_length:
#             next_answer = dialogues[next_answer_index][1][:n_characters_same]
#             if current_answer == next_answer:
#                 return False
#     return True
"""

def check_conversation_length(conv, qatuples):
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

def check_conversation_for_text_from_examples(conv):
    """Checks if certain strings from the few-shot examples appear in the conversation"""
    strings_to_check_for = [
        "her lipstick-colored lips",
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

def check_each_question_contains_q_from_tuples(conv, qatuples, n):
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

def check_for_unintended_repeated_quotes(dialogues, qatuples, n_characters_shared):
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

    for i in range(
        2, len(dialogues), 2
    ):  # Answers are at even indices, starting from 2
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

def call_all_processors(multiturn_conversation, qatuples):
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
                "output_text": ("OUTPUT_TEXT",)
            },
            "hidden": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "display_output_text"

    output_node = True

    CATEGORY = "helper_functions"

    def display_output_text (self, output_text):

        output_text = output_text['OUTPUT_TEXT']
        print(output_text)

        return {"ui": {"text": output_text}}

#TODO This might cause problems if the nodes its hooked up to are run simultaneously, as it will be calling the same model with two different inputs at the same time.
#TODO Set up overrides in nodes with INITIALIZED_MODEL inputs
#TODO Investigate the limitations of partial functions.
#TODO Finish documenting the function
class InitializeLlm: 
    """
    This function initializes a loaded LLM with the given parameters settings.
    :param model: The loaded LLM  (external)
    :param max_tokens: The maximum number of tokens that can be generated in a single run.
    :param max_token_length: The maximum token length for a chunk of sentences
    :param stop_arg: The specified stop tokens, usually ["</s>", "# Input:"] 
    :param echo_arg: 
    :param temperature_arg: The specified temperature. Scales token probabilities so that 2nd, 3rd, etc most-likely tokens have a greater likelihood of being selected.
    :param top_k_arg: 
    :param top_p_arg: 
    :param min_p_arg: 
    :param seed_arg: 
    :param llm_type: 
    :param override_llm_presets_in_connected_node:
    :return initialized_model: an initialized model with the given parameter generation settings.
    :return override_llm_presets: a boolean indicating whether these LLM presets take precedence over the presets in the connected node.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "max_tokens_arg": ("INT", {"default": 6000, "min":1, "max":100000, "step":1}),
                "stop_arg": ("STRING", {"default": '["</s>", "# Input:"]'}),
                "echo_arg": (["True", "False"],),
                "temperature_arg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}), # Default is 0, which is deterministic e.g. the most likely token is always selected.
                "top_k_arg": ("INT", {"default": 40, "min":1, "max":1000, "step":1}), # Default is 40, which means only the top 40 token probabilities are considered for selection.
                "top_p_arg": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 20.0, "step": 0.01}), # Default is 1, which is off (???)
                "min_p_arg": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}), # Default is 0, which is off (???)
                "seed_arg": ("INT", {"default": 0, "min": 0, "max":0xffffffffffffffff, "step":1}), # Default is 0 i.e. the seed is randomly generated.
                "override_llm_presets_in_connected_node": (["False", "True"],), # If this is False, none of the above options actually do anything.
            }
        }
    RETURN_TYPES = ("INITIALIZED_MODEL","OVERRIDE_LLM_PRESETS_CHOICE",)

    FUNCTION = "initialize_llm"
    CATEGORY = "helper_functions"

    def initialize_llm(self, model, max_tokens_arg, stop_arg, echo_arg, temperature_arg, top_k_arg, top_p_arg, min_p_arg, seed_arg, override_llm_presets_in_connected_node):

        try:
            if override_llm_presets_in_connected_node == "True":
                override_llm_presets = True
                # This creates a Partial Function, which is essentially the same function but with specified arguments for it fixed in place.
                initialized_model = partial(model,  # First argument of partial is the function we want to fix some of the arguments for.
                    max_tokens=max_tokens_arg,
                    stop=stop_arg,
                    echo= True if echo_arg == "True" else False,
                    temperature=temperature_arg,
                    top_k=top_k_arg,
                    top_p=top_p_arg,
                    seed=None if seed_arg == 0 else seed_arg,
                    min_p=min_p_arg,
                )
            else:
                # This just lets the presets in the connected node take over, and is essentially a dummy variable.
                override_llm_presets = False
                initialized_model = partial(model,
                    mirostat_mode=0 # Fuck you, mirostat is a worthless option anyways!
                )

            logger.info(f"LLM initialization successful. override_llm_presets is currently set to: {override_llm_presets}")
            time.sleep(3)

        except Exception as e:
            logger.exception(f"An Exception occurred in initialize_llm function in InitializeLlm class: {e}")

        return (initialized_model, override_llm_presets,)


#TODO: What about CPU-based models with VRAM off-loading? 
# Also, this produces a recursion loop. Fix it somehow! Maybe by a global variable???
# Also, this needs it's own clickable button widget.Also, needs to be checked to see if it works at all.
class PurgeLlmFromRamOrVram: 
    """
    This function purges the LLM from RAM or VRAM, since llama-cpp-python has a memory leak problem.
    :param model: The loaded LLM  (external)
    :param purge_trigger: A boolean input that triggers the purge automatically.
    :return: NONE: The output is a clean RAM and/or VRAM cache!
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
    RETURN_TYPES = ("RELOAD_MODEL")
    FUNCTION = "purge_llm_from_ram_or_vram"

    OUTPUT_NODE = True

    CATEGORY = "helper_functions"

    def purge_llm_from_ram_or_vram(self, purge_trigger=False):
        if purge_trigger:
            release_memory(self.loaded_model)
            # del self.loaded_model
            reload_model = True
        
        return (None, reload_model,)

class WriteOutputToFile:
    """
    This function outputs a text or json file to the output directory.
    :param output: the file to be output (external)
    :param filename_prefix: the prefix you want to stick onto the front of the output file.
    :return: NONE: the output file is the return for this function!
    TODO: Get this so that it can save multiple output texts in succession. 
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "filename_prefix": ("STRING", {"default": "augmentoolkit"},),
                "file_output_type": (["txt","json"],),
            },
            "optional": {
                "output_text": ("OUTPUT_TEXT",),
                "output_tuple": ("TUPLE", {"forceInput": True}),
            },
        }
    RETURN_TYPES = ()
    FUNCTION = "write_output_to_file"

    OUTPUT_NODE = True

    CATEGORY = "helper_functions"

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
                logger.error(f"Error: write_output_to_file function can only take a output_tuple or output_text as an argument, not both.")

            logger.info(f"Output written to {file_path}")

            results.append({
                "filename": f"{filename_prefix}_{uuid}",
                "folder": file_path,
                "type": self.type
            })

        except Exception as e:
            logger.exception(f"An Exception occured in write_output_to_file function when writing output to {file_path}: {e}")

        return { "ui": { "results": results } }

NODE_CLASS_MAPPINGS = {
    #Helper Functions
    "DisplayOutputText": DisplayOutputText,
    #"ExtractConversation": ExtractConversation,
    "InitializeLlm": InitializeLlm,
    "PurgeLlmFromRamOrVram": PurgeLlmFromRamOrVram,
    #"GroupByText": GroupByText,
    "WriteOutputToFile": WriteOutputToFile
}

NODE_DISPLAY_NAME_MAPPINGS = {
    #Helper Functions
    "DisplayOutputText": "Display Output Text",
    #"ExtractConversation": "Extract Conversation",
    "InitializeLlm": "Initialize LLM",
    "PurgeLlmFromRamOrVram":"Purge Loaded LLM from RAM or VRAM",
    #"GroupByText": "Group by Text",
    "WriteOutputToFile": "Write Output to File"
}


#read_json_files_info

#############################
#### OTHER ODDS AND ENDS ####
#############################

RP_MODEL = "./rp_model"  # model used for RP tasks, probably going to use Sao10K/Euryale-1.3-L2-70b
# LOGICAL_MODEL = "./logical_model/airoboros-l2-13b-3.1.1.Q8_0.gguf" # model used for decision-making and base question generation (should be "smart")

LOGICAL_MODEL = "./logical_model/flatorcamaid-13b-v0.2.Q8_0.gguf"  # model used for decision-making and base question generation (should be "smart")




















