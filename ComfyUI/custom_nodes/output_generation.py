import asyncio
import glob
import io
import inspect
import itertools
import json
import logging
import os
import sys
import random
import re
import string
import time
import traceback
import uuid

from llama_cpp import Llama, LlamaGrammar
from math import ceil
from tqdm import tqdm
from typing import List, Tuple
from datetime import datetime

script_dir = os.path.dirname(os.path.realpath(__file__))
custom_nodes_path = os.path.join(script_dir, "ComfyUI", "custom_nodes")

sys.path.insert(0, custom_nodes_path)

from custom_nodes import helper_functions, grammars, output_validation
#from grammars import Grammars
#from custom_nodes.output_validation import extract_first_words
from custom_nodes.helper_functions import format_external_text_like_f_string, write_output_to_file, call_all_processors, check_for_unintended_repeated_quotes, check_each_question_contains_q_from_tuples, check_conversation_for_text_from_examples, check_conversation_length, check_for_repeated_dialogue_answers, compare_answers_with_qatuples, extract_conversation, has_sequential_chars, load_external_prompt_and_grammar
from custom_nodes.logger import logger
import folder_paths


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
        logger.exception(f"An Exception occurred when creating the prompt dictionary object: {e} ")

GRAMMAR_DICT = {}
for file_name in folder_paths.get_filename_list("grammars"):
    try:
        file_path = folder_paths.get_full_path("grammars", file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            key = os.path.splitext(file_name)[0]
            GRAMMAR_DICT[key] = file.read()
    except Exception as e:
        logger.exception(f"An Exception occurred when creating the grammar dictionary object: {e} ")

##############################
###### GLOBAL VARIABLES ######
##############################

#LOGICAL_MODEL = "./logical_model/flatorcamaid-13b-v0.2.Q8_0.gguf"  # model used for decision-making and base question generation (should be "smart")
#LARGE_LOGICAL_MODEL = "./logical_model/airoboros-l2-70b-3.1.2.Q4_K_M.gguf"

ASSISTANT_MODE = True  # change to true if you want all conversations to be with an "AI language model" and not characters. Useful for more professional use cases.

DEBUG_MODE = True

DOUBLE_CHECK_COUNTER = 3  # Set to 1 to check outputs only once; set to 2 to check twice; set to 3 to check thrice, etc. Set to 0 to break everything in vet_question_loop() and elsewhere. Set to -1 and cause the universe to implode?

REARRANGEMENTS_TO_TAKE = 3  # How many of the possible permutations of tuples in a group to take and make multiturn convs out of. Adjust higher to get more data out of less text, but it might be a bit repetitive. NOTE your eval loss will be basically worthless if you aren't careful with how you shuffle your dataset when you're about to train.

TEXT_MANUALLY_CLEANED = False  # If you've manually cut out all the parts of the text not worthy for questions, you can skip the first LLM step. NOTE I might actually recommend doing this if your source text is small, given how permissive the filtering prompt is; it really only disallows metadata.

# These do nothing at the moment. May change later - KR
source_texts = [
    "Simple Sabotage, by the Office of Strategic Services, published 1944.txt",
    "Principles of Chemistry, by Demitry Mendeleev, published 1897.txt",
]

NAMES = [  # Replaces "Albert" in scenarios. Needs to be western male names to avoid pronoun and setting inconsistencies).
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

##############################
#### INDIVIDUAL FUNCTIONS ####
##############################
# Note: This is primarily for nodes that have so many functions within them that it's just easier to call them from outside the class.
# Moving them outside the node also allows nodes to be divided into "basic" and "advanced", making them more flexible for users.

# Note: I added in try-except blocks to the retry loops, specifically functions like make_regenerate_answer_constrain_to_text_plan.
# If this wasn't intended, please change it back.
# Note: ALL of these prompts rely on the format_external_text_like_f_string function in order to import their prompts. 
# Be VERY careful about messing with it as it could break EVERYTHING.

#### RANDOM FUNCTIONS ####

#TODO: Alphabetize these for ease of access.
#TODO Add in LLM setting override options into these. Might make this a global variable, but then it would override EVERYTHING, not just connected nodes.
# Should be useful for prompt debugging purposes and general experimentation once it's implemented.
#TODO Assistant mode might also need a bit of work to make it work for node-based usage. Making it a global variable makes it inflexible.
#TODO Refactor functions for consistency in format.

def combine_traits(personality_matrix):  # GPT-generated

    # Using itertools.product to generate all possible combinations
    combinations = itertools.product(*personality_matrix)

    # Joining each combination into a single string
    combined_traits = [
        "\n".join(combination).strip().replace("\n\n", "\n")
        for combination in combinations
    ]

    return combined_traits


def create_character_card_many_tuples(qatuples, plan, instructions, initialized_model, cheap_mode=False):  # Use cheap mode if you don't have the compute power to crank up the context to 8k using RoPE
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.
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
        cot_prompt, character_card_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "character_card_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

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
        #logger.info(f"\n*** Completion for 'create_character_card_plan_many_tuples' function ***\n{completion}\n*** Completion for 'create_character_card_plan_many_tuples' function ***\n")
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
    #logger.info(f"\nGeneration for 'create_character_card_many_tuples' function ***\n{generation}\n *** Generation for 'create_character_card_many_tuples' function")

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
            instructions_string = special_instructions_prototype(n=1)
    else:
            instructions_string = special_instructions_prototype2(n=1)

    prompt_content = {
        "qatuples": qatuples,
        "instructions_string_strip": instructions_string.strip(),
        "format_qatuples_qatuples": format_qatuples(qatuples),
    }

    # Load the prompt and the grammar.
    try:
        cot_prompt, character_card_plan_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "character_card_plan_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

    #logger.info(f"\n*** create_character_card_plan_many_tuples cot_prompt ***\nqatuples:{cot_prompt}\n*** create_character_card_plan_many_tuples cot_prompt ***\n")
    #time.sleep(5)

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
        #logger.info(f"\n*** Completion for 'create_character_card_plan_many_tuples' function ***\n{completion}\n*** Completion for 'create_character_card_plan_many_tuples' function ***\n")
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
    #logger.info(f"\nGeneration for 'create_character_card_plan_many_tuples' function ***\n{generation}\n *** Generation for 'create_character_card_plan_many_tuples' function")

    return generation, instructions_string, completion


# TODO Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just return an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah
# Actually instead of the scenario being a blank string, I'll have it describe a text conversation between a helpful AI assistant and a user. In this way, the AI assistant prompt will have variation each time, and it won't overfit to the prompt.
def create_scenario_plan(qatuple, character, initialized_model, assistant_mode=False):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """

    # removing the text makes this much better

    # The problem: because the scenario plan differed slightly, the question differed slightly. Because the question differed slightly, the answer differed slightly. Because the answer differed slightly, the answer was incomplete.
    if assistant_mode:
        return """"""
    prompt_content = {
        "qatuple": qatuple,
        "character": character,
    }

    # Load the prompt and the grammar.
    try:
        cot_prompt, scenario_plan_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "scenario_plan_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

    # Even if the example does a justified clever trick, the model imitating it may fuck up the trick. So try to avoid complex things that aren't needed for the task in examples, like the "just how much have you dug" colloquialization
    completion = initialized_model(
        cot_prompt,
        max_tokens=4000,
        stop=["</s>", "# Input:"],
        echo=True,
        grammar=scenario_plan_grammar,
        temperature=0.2,
    )["choices"][0]["text"]
    # print("COMPLETION:\n\n----------------------")
    # # print(completion)
    # print("\n------------------")

    # Extract plan
    response_pattern = re.compile(
        r"Scenario plan \(be creative, and make sure all characters present fit in with the setting\):([\s\S]*)",
        re.IGNORECASE | re.DOTALL,
    )
    generation = response_pattern.search(completion).group(1)
    # print("GENERATION:\n\n-------------------\n\n", generation)

    return generation


def create_thought_plan_many_tuples(qatuples, character, initialized_model):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. 
    The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """

    # use regex to extract charname
    charname = extract_name(character)
    thought_plans = []
    for tuple in qatuples:
        thought_plans.append(create_thought_plan(tuple, character, initialized_model))

    ret = ""
    for idx, plan in enumerate(thought_plans):
        ret += f"Plan for Question {idx}:\n{plan}\n\n"

    return "\n\n".join(thought_plans)


# Answer vetting
# For now, this checks answer relevancy too. The danger with abstracting answer relevancy into a separate step is that anything which relies on knowledge that is obviously mentioned in the text already up until this point, will get screwed
def ensure_answer_consistent(qatuple, conv, initialized_model, permissive_mode=True):
    """
    permissive_mode: turn off if you want a single usage of the word "inconsistent" anywhere in the message to flag the whole thing as inconsistent. Prevents errors where an inconsistency happens way early in the answer, but the model forgets about it during its final judgement; but enables the error where the model mentions that something is "not entirely inconsistent" or similar, which is surprisingly common.
    """
    retries = 0

    # It's expensive to regen a conversation; so we check very thoroughly, and use a two-shot example. "Permissive mode" recommended

    # NOTE: I don't know what kind of errors this part of the pipeline will run into most often, so I don't really know what examples to feed it to guard it with. 
    # Come back to it once I have tested it more.
    while retries <= 4:

        prompt_content = {
            "qatuple": qatuple,
            "conv": conv,
        }

        # Load the prompt and the grammar.
        try:
            decision_prompt, ensure_answer_consistent_grammar = load_external_prompt_and_grammar('ensure_answer_consistent', "ensure_answer_consistent_grammar", prompt_content)
        except Exception as e:
            logger.exception(f"An Exception occurred in 'ensure_answer_consistent' function while trying to import its prompt and grammar: {e}")
            break

        # print("DEBUG\n\n" + decision_prompt)
        try:
            completion = initialized_model(
                decision_prompt,
                max_tokens=4000,
                stop=["</s>", "# Input:"],
                echo=True,
                grammar=ensure_answer_consistent_grammar,
                temperature=0.2,
            )["choices"][0]["text"]

            completion_pattern = re.compile(
                r"Reasoning and thought process \(the conversation's answer must match the provided answer, unsummarized and unsimplified\):([\s\S]*)", 
                re.DOTALL,
            )
            response = completion_pattern.search(completion).group(1).strip()
            # print("DEBUG\n\n")
            # print(completion)

            if permissive_mode:
                determination_pattern = re.compile(r"Final Judgement:([\s\S]*)", re.DOTALL)
                determination = determination_pattern.search(response).group(1).strip()
            else:
                determination = response

            logger.info(f"\n\nDETERMINATION:\n------\n{determination}\n---------\n")

            if "inconsistent" in determination.lower():
                return (False, response)
            elif "consistent" in determination.lower():
                return (True, response)
            else:
                retries += 1

        except Exception as e:
            retries += 1
            logger.exception(f"Something went catastrophically wrong with this one: {e} Investigate! Here's the completion:\n{completion}")
            continue

# Answer vetting
# For now, this checks answer relevancy too. The danger with abstracting answer relevancy into a separate step is that anything which relies on knowledge that is obviously mentioned in the text already up until this point, will get screwed
# NOTE this prompt right now VERY MUCH struggles to follow its actual format; but it still mostly works
def ensure_multiple_answers_consistent(qatuples, conv, initialized_model, permissive_mode=True):
    """
    permissive_mode: turn off if you want a single usage of the word "inconsistent" anywhere in the message to flag the whole thing as inconsistent. Prevents errors where an inconsistency happens way early in the answer, but the model forgets about it during its final judgement; but enables the error where the model mentions that something is "not entirely inconsistent" or similar, which is surprisingly common.
    """
    retries = 0
    character_name = extract_name(conv[1])
    # It's expensive to regen a conversation; so we check very thoroughly, and use a two-shot example. "Permissive mode" recommended

    # NOTE: I don't know what kind of errors this part of the pipeline will run into most often, so I don't really know what examples to feed it to guard it with. Come back to it once I have tested it more.

    # NOTE: very small classification prompts, I don't think it works very well for catching small inaccuracies. We need the large, step-by-step analysis.

    # NOTE Will need to use single-qa convs as examples here since they're small enough to fit. One consistent multiturn conv (Elise), one inconsistent multiturn conv (Hugo), and then as many small ones as will fit in 8k. Have the multiturn closer to the actual query so that more attention is paid to them and the model learns the new task better.

    # NOTE Introduction to Practicing Chemical Science does not exist; this is more stuff from principles of chemistry named otherwise to avoid biasing the outputs more than can be helped
    # Consider removing the "conversational fluff" bit of the prompt. It's not really necessary? maybe?
 
    context = {
        "conv": conv,
        "format_qatuples_qatuples": format_qatuples(qatuples),
        "character_name": character_name,
    }

    # Load the prompt.
    try:
        decision_prompt, _ = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "ensure_multiple_answers_consistent_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt: {e}")

    while retries <= 4:

        try:
            completion = initialized_model(
                decision_prompt,
                max_tokens=12000,
                stop=["</s>", "# Input:"],
                echo=True,
                # grammar=ensure_multiple_answers_consistent_grammar,#temperature=0.2
                temperature=0.5,  # min p settings, too inconsistent
                top_k=0,
                top_p=1,
                min_p=0.6,
            )["choices"][0]["text"]

            #print("DEBUG\n\n")
            # print(completion)
            completion_pattern = re.compile(
                r"Response \(the conversation's answer must match the provided answer, unsummarized and unsimplified; added questions that are rhetorical or part of the plot \(such as 'would you like to get coffee'\) are acceptable\):([\s\S]*)", 
                re.DOTALL,
            )
            response = completion_pattern.search(completion).group(1).strip()
            # print(completion)

            if permissive_mode:
                determination_pattern = re.compile(
                    r"Final Judgment:([\s\S]*)", 
                    re.IGNORECASE
                )
                determination = determination_pattern.search(response).group(1).strip()
            else:
                determination = response

            logger.info(f"\n\nDETERMINATION:\n------\n{determination}\n---------\n")

            if "inconsistent" in determination.lower():
                return (False, response)
            elif "consistent" in determination.lower():
                return (True, response)
            else:
                retries += 1

        except Exception as e:
            retries += 1
            logger.exception(f"Something went catastrophically wrong with this one: {e} \nInvestigate! Here's the completion:\n{completion}")
            time.sleep(5)
            continue


def extract_author_name(title):
    pattern = re.compile(r"\b(?:by|By)\s+([^,]+),")
    match = re.search(pattern, title)
    if match:
        author_name = match.group(1)
    else:
        author_name = [False]
    return author_name[0]  # first letter of Author name


def extract_capital_letters(input_string):
    capital_letters = []
    for char in input_string:
        if char.isupper():
            capital_letters.append(char)
    return capital_letters


def extract_first_words(character_name, text):
    # Regular expression pattern to extract first word after the character's name
    pattern = fr"{character_name}: \"(\w+)"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    return matches


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
        logger.warning(f"No name found using extract_name function on {str}.")


def extract_steps(text, steps=[5, 6]):  # GPT-generated
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


def make_id():
    return str(uuid.uuid4())


def make_regenerate_answer_constrain_to_text_plan(qatuple, dissenting_reasoning, initialized_model):
    retries = 0
    prompt_content = {
        "strip_steps_dissenting_reasoning": strip_steps(dissenting_reasoning),
        "qatuple": qatuple,
    }

    # Load the prompt and the grammar.
    try:
        decision_prompt, answer_constrain_to_text_plan_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "answer_constrain_to_text_plan_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

    while retries < 5:

        try:
            start_time = time.time()
            completion = initialized_model(
                decision_prompt,
                max_tokens=3000,
                stop=["</s>", "# Input:"],
                echo=True,
                grammar=answer_constrain_to_text_plan_grammar,
                temperature=0.2,
            )["choices"][0]["text"]

            end_time = time.time()
            logger.info(f"Completion took {(end_time - start_time) / 60} minutes to complete.")
            logger.info(f"Completion for 'make_regenerate_answer_constrain_to_text_plan' function generated. Extracting correction pattern...")

            # print("DEBUG\n\n")
            # print(completion)
            completion_pattern = re.compile(
                r"Reasoning and thought process:([\s\S]*)", re.DOTALL
            )
            correction = completion_pattern.search(completion).group(1)
            logger.info(f"*** Corretion for 'make_regenerate_answer_constrain_to_text_plan' function.*** \n{correction} \n*** Corretion for 'make_regenerate_answer_constrain_to_text_plan' function.***")

            return correction

        except Exception as e:
            retries += 1
            logger.exception(f"Something went catastrophically wrong with this one: {e} Investigate! Here's the completion:\n{completion}")
            continue


def select_random_capital(exclusions):
    # Create a list of capital letters excluding the ones in the exclusions list
    capitals = [letter for letter in string.ascii_uppercase if letter not in exclusions]

    # Select a random capital letter from the filtered list
    if capitals:
        return random.choice(capitals)
    else:
        return "No available capital letters to choose from"


def single_turn_conversation(qatuple, character, scenario, thought_plan, scenario_plan, initialized_model, assistant_mode=False,):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """

    extra_info = extract_steps(scenario_plan)

    # Load the grammar.
    try:
        _ , single_turn_conversation_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "single_turn_conversation_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its grammar: {e}")

    if assistant_mode:  # TODO
        prompt_content = {
            "character": character,
            "scenario": scenario,
            "thought_plan": thought_plan,
            "qatuple": qatuple,
        }

        # Load the assistant prompt.
        try:
            cot_prompt , _ = load_external_prompt_and_grammar(f"{inspect.currentframe().f_code.co_name}_assistant_mode", "single_turn_conversation_grammar", prompt_content)
        except Exception as e:
            logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its assistant prompt: {e}")

    else:
        prompt_content = {
            "character": character,
            "scenario": scenario,
            "thought_plan": thought_plan,
            "extra_info": extra_info,
            "qatuple": qatuple,
        }

        # Load the regular prompt.
        try:
            cot_prompt , _ = load_external_prompt_and_grammar(f"{inspect.currentframe().f_code.co_name}", "single_turn_conversation_grammar", prompt_content)
        except Exception as e:
            logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt: {e}")


    # Higher temp definitely makes the writing better, but highly predisposes it to not use only info in the test. ): I want min p goddamn it
    # Note: performance degrades rapidly if you put more than one sentence in a pre-prompt parentheses thing

    start_time = time.time()
    logger.info(f"Generating 'single_turn_conversation' completion...")

    completion = initialized_model(
        cot_prompt,
        max_tokens=4096,
        stop=["</s>", "# Input:"],
        echo=True,
        grammar=single_turn_conversation_grammar,
        temperature=0.2,
    )["choices"][0]["text"]

    end_time = time.time()
    logger.info(f"Done! Completion took {(end_time - start_time) / 60} minutes to generate.")
    logger.info(f"Completion for 'single_turn_conversation' function generated. Extracting response pattern...")

    # Extract plan
    response_pattern = re.compile(
        r"Conversation that answers the provided question \(first, the secondary character will ask the question; then, the primary character will answer it\):([\s\S]*)",
        re.IGNORECASE | re.DOTALL,
    )
    generation = response_pattern.search(completion).group(1)
    logger.info(f"*** Response for 'single_turn_conversation' function ***\n{generation}\n *** Response for 'single_turn_conversation' function ***")

    return generation


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


# Custom version of the special instructions, with Big Five Personality traits and CAT-Personality Disorder Scale traits added in.
# I've got plans for this one...
# TODO: Make the traits externally loadable, so I can fuck around with them in notepad or word without potentially fucking up the code.
# TODO: Add a help menu to explain to users what the fuck these schemas are.
def special_instructions_prototype_2(personality_trait_schema, num_of_traits):
    # This is my function. There are many like it, but this one is MINE!!! - KR

    # Big Five Personality Traits. See: https://en.wikipedia.org/wiki/Big_Five_personality_traits
    big_five_traits = [
        "The character has High Conscientiousness. Conscientiousness describes a person’s ability to regulate impulse control in order to engage in goal-directed behaviors. It measures elements such as control, inhibition, and persistence of behavior. Those high in conscientiousness can be described as organized, disciplined, detail-oriented, thoughtful, and careful. They also have good impulse control, which allows them to complete tasks and achieve goals.",
        "The character has Low Conscientiousness. Conscientiousness describes a person’s ability to regulate impulse control in order to engage in goal-directed behaviors. It measures elements such as control, inhibition, and persistence of behavior. Those low in conscientiousness may struggle with impulse control, leading to difficulty in completing tasks and fulfilling goals. They tend to be more disorganized and may dislike too much structure. They may also engage in more impulsive and careless behavior.",
        "The character has High Agreeableness. Agreeableness refers to how people tend to treat relationships with others, and focuses on people’s orientation and interactions with others. Those high in agreeableness can be described as soft-hearted, trusting, and well-liked. They are sensitive to the needs of others and are helpful and cooperative. People regard them as trustworthy and altruistic.",
        "The character has Low Agreeableness. Agreeableness refers to how people tend to treat relationships with others, and focuses on people’s orientation and interactions with others. Those low in agreeableness may be perceived as suspicious, manipulative, and uncooperative. They may be antagonistic when interacting with others, making them less likely to be well-liked and trusted.",
        "The character has High Extraversion. Extraversion reflects the tendency and intensity to which someone seeks interaction with their environment, particularly socially. It encompasses the comfort and assertiveness levels of people in social situations. Those high in extraversion are generally assertive, sociable, fun-loving, and outgoing. They thrive in social situations and feel comfortable voicing their opinions. They tend to gain energy and become excited from being around others.",
        "The character has Low Extraversion. Extraversion reflects the tendency and intensity to which someone seeks interaction with their environment, particularly socially. It encompasses the comfort and assertiveness levels of people in social situations. Those low in extraversion are often referred to as introverts. These people tend to be more reserved and quieter. They prefer listening to others rather than needing to be heard. Introverts often need periods of solitude in order to regain energy as attending social events can be very tiring for them. Of importance to note is that introverts do not necessarily dislike social events, but instead find them tiring.",
        "The character has High Openness to Experience. Openness to experience refers to one’s willingness to try new things as well as engage in imaginative and intellectual activities. It includes the ability to “think outside of the box.” Those high in openness to experience are perceived as creative and artistic. They prefer variety and value independence. They are curious about their surroundings and enjoy traveling and learning new things.",
        "The character has Low Openness to Experience. Openness to experience refers to one’s willingness to try new things as well as engage in imaginative and intellectual activities. It includes the ability to “think outside of the box.” Those low in openness to experience prefer routine. They are uncomfortable with change and trying new things, so they prefer the familiar over the unknown. As they are practical people, they often find it difficult to think creatively or abstractly.",
        "The character has High Neuroticism. Neuroticism describes the overall emotional stability of an individual through how they perceive the world. It takes into account how likely a person is to interpret events as threatening or difficult. It also includes one’s propensity to experience negative emotions. Those high in neuroticism often feel anxious, insecure and self-pitying. They are often perceived as moody and irritable. They are prone to excessive sadness and low self-esteem.",
        "The character has Low Neuroticism. Neuroticism describes the overall emotional stability of an individual through how they perceive the world. It takes into account how likely a person is to interpret events as threatening or difficult. It also includes one’s propensity to experience negative emotions. Those low in neuroticism are more likely to calm, secure and self-satisfied. They are less likely to be perceived as anxious or moody. They are more likely to have high self-esteem and remain resilient."
    ]

    # CAT-Personality Disorder Scales Static form (CAT-PD-SF,v1.1) See: https://ipip.ori.org/newCAT-PD-SFv1.1Keys.htm
    # These have been changed by converting them to third-person (singular they), inverting reverse-keyed questions, and changing negatives to antonyms if possible (.e.g does not trust -> distrusts).
    # Some have been heavily modified, so using this as an actual personality test for real people is not recommended.
    cat_personality_disorder = [
        [# Affective Lability (6 items; Comm α = .83, Pat α = .86)
            "The character has frequent mood swings.",
            "The character loses control over their behavior when they're emotional.",
            "The character has unpredictable emotions and moods.",
            "The character overreacts to every little thing in life.",
            "The character lacks coping skills.", # Revised due to original being reverse-keyed.
            "The character is hot-headed when stressed out." # Revised due to original being reverse-keyed.
        ],
        [# Anger (6 items; Comm α = .83, Pat α = .85)
            "The character gets angry easily.",
            "The character often feels overwhelmed with rage.",
            "The character gets irritated easily.",
            "The character has a violent temper.",
            "The character is easily annoyed.", # Revised due to original being reverse-keyed.
            "The character lets little things anger them." # Revised due to original being reverse-keyed.
        ],
        [# Anhedonia (6 items; Comm α = .84, Pat α = .89)
            "The character finds nothing that excites them.",
            "The character feels that nothing seems to make them feel good.",
            "The character is an apathetic person.",
            "The character has trouble getting interested in things.",
            "The character seldom has a lot of fun.", # Revised due to original being reverse-keyed.
            "The character is an unenergetic person." # Revised due to original being reverse-keyed.
        ],
        [# Anxiousness (7 items; Comm α = .83, Pat α = .85)
            "The character feels that their anxiety overwhelms them.",
            "The character is nervous or tense most of the time.",
            "The character panics easily.",
            "The character feels that their worry and anxiety is out of control.",
            "The character is generally a fearful person.",
            "The character is easily startled.",
            "The character always worries." # Revised due to original being reverse-keyed.
        ],
        [# Callousness (7 items; Comm α = .85, Pat α = .83)
            "The character cares little about others.", # Revised due to original being reverse-keyed.
            "The character is an uncaring person.",
            "The character is a cold-hearted person.",
            "The character is unconcerned about how their actions affect others.",
            "The character is indifferent to other's needs.",
            "The character is an unsympathetic person.",
            "The character is indifferent to the feelings of others."
        ],
        [# Cognitive Problems (8 items; Comm α = .82, Pat α = .88)
            "The character frequently get things mixed up in their head.",
            "The character often feels like their thoughts make no sense.",
            "The character often spaces out and lose track of what's going on.",
            "The character often has disorganized thoughts.",
            "The character is easily disoriented.",
            "The character easily loses their train of thought.",
            "The character has a poor memory for things they've done throughout the day.", # Revised due to original being reverse-keyed.
            "The character struggles to formulate ideas clearly." # Revised due to original being reverse-keyed.
        ],
        [# Depressiveness (6 items; Comm α = .88, Pat α = .88)
            "The character tends to feel very hopeless.",
            "The character is sad most of the time.",
            "The character generally focuses on the negative side of things.",
            "The character dislikes themselves.",
            "The character seldom looks at the bright side of life.", # Revised due to original being reverse-keyed.
            "The character frequently feels depressed." # Revised due to original being reverse-keyed.
        ],
        [# Domineering (6 items; Comm α = .83, Pat α = .84)
            "The character bosses people around.",
            "The character likes having authority over others.",
            "The character insists that others do things their way.",
            "The character makes demands on others.",
            "The character has a strong need for power.",
            "The character is known as a controlling person."
        ],
        [# Emotional Detachment (7 items; Comm α = .82, Pat α = .86)
            "The character has difficulty expressing their feelings.",
            "The character thinks it's best to keep their emotions to themselves.",
            "The character is guarded about their feelings.", # Revised due to original being reverse-keyed.
            "The character is bad at describing the emotions they feel throughout the day.",
            "The character has difficulty showing affection.",
            "The character has difficulty describing their feelings.", # Revised due to original being reverse-keyed.
            "The character is emotionally reserved."
        ],
        [# Exhibitionism (6 items; Comm α = .82, Pat α = .83)
            "The character loves to be the center of attention.",
            "The character likes to stand out in a crowd.",
            "The character is likely to show off if they get the chance.",
            "The character uses their looks to get what they want.",
            "The character enjoys flirting with complete strangers.",
            "The character enjoys being in the spotlight." # Revised due to original being reverse-keyed.
        ],
        [# Fantasy Proneness (6 items; Comm α = .82, Pat α = .83)
            "The character sometimes gets lost in their daydreams.",
            "The character sometimes has fantasies that are overwhelming.",
            "The character sometimes finds themselves in a trance-like state without trying.",
            "The character feels like their imagination can run wild.",
            "The character is sometimes so preoccupied with their own thoughts that they fail to realize others are trying to speak to them.",
            "The character sometimes has extremely vivid pictures in their head."
        ],
        [# Grandiosity (7 items; Comm α = .85, Pat α = .81)
            "The character thinks they deserve special treatment from others.",
            "The character thinks they should get special privileges.",
            "The character believes that they are better than others.",
            "The character questions why they should have to wait in lines like others.",
            "The character feels that others are beneath them.",
            "The character believes that they are always right.",
            "The character treats people as inferiors."
        ],
        [# Health Anxiety (7 items; Comm α = .84, Pat α = .84)
            "The character worries a lot about catching a serious illness.",
            "The character is prone to complaining about their health.",
            "The character is often concerned about diseases they might have.",
            "The character is afraid that their life will be cut short by illness.",
            "The character feels like they have medical problems that their doctors fail to understand.",
            "The character worries about their health.",
            "The character thinks that they are in poor medical condition." # Revised due to original being reverse-keyed.
        ],
        [# Hostile Aggression (8 items; Comm α = .82, Pat α = .87)
            "The character is often out for revenge.",
            "The character is excited to inflict pain on others.",
            "The character gets even with others.",
            "The character hurts people.",
            "The character will spread false rumors as a way to hurt others.",
            "The character is ready to hit someone when they get angry.",
            "The character likes to start fights.",
            "The character enjoys a good brawl."
        ],
        [# Irresponsibility (7 items; Comm α = .82, Pat α = .85)
            "The character neglects their duties.",
            "The character seldom follows through with their plans.", # Revised due to original being reverse-keyed.
            "The character seldom keeps their appointments.", # Revised due to original being reverse-keyed.
            "The character is a pretty unreliable person.", # Revised due to original being reverse-keyed.
            "The character avoids responsibilities.",
            "The character cannot be counted on to get things done.",
            "The character is an undependable person."
        ],
        [# Manipulativeness (6 items; Comm α = .88, Pat α = .85)
            "The character takes advantage of others.",
            "The character cheats to get ahead.",
            "The character likes to trick people into doing things for them.",
            "The character deceives people.",
            "The character has exploited others for their own gain.",
            "The character is a dishonest person." # Revised due to original being reverse-keyed.
        ],
        [# Mistrust (6 items; Comm α = .83, Pat α = .88)
            "The character feels like people often are out to get something from them.",
            "The character feels that others are out to get them.",
            "The character believes that, sooner or later, people always let you down.",
            "The character suspects hidden motives in others.",
            "The character believes that people are basically dishonest and evil.", # Revised due to original being reverse-keyed.
            "The character is pretty distrusting of others' motives." # Revised due to original being reverse-keyed.
        ],
        [# Non-Perseverance (6 items; Comm α = .83, Pat α = .88)
            "The character quickly loses interest in the tasks they start.",
            "The character has difficulty keeping their attention on a task.",
            "The character is easily distracted.",
            "The character quits tasks as soon as they get bored.",
            "The character fails to finish what they start.", # Revised due to original being reverse-keyed.
            "The character is quick to quit when the going gets tough."
        ],
        [# Non-Planfulness (6 items; Comm α = .82, Pat α = .84)
            "The character does things without thinking of the consequences.",
            "The character acts without planning.",
            "The character jumps into things without thinking.",
            #"The character is not a firm believer in thinking things through.", # Revised due to original being reverse-keyed.
            "The character is a firm believer in following their instincts",
            "The character makes careless choices.", # Revised due to original being reverse-keyed.
            "The character prefers to 'live in the moment' rather than plan things out."
        ],
        [# Norm Violation (7 items; Comm α = .83, Pat α = .84)
            "The character has always been a rule-breaker.",
            "The character gets in trouble with the law.",
            "The character is a law-breaking citizen.", # Revised due to original being reverse-keyed.
            "The character disrespects authority.", # Revised due to original being reverse-keyed.
            "The character has a rebellious side that gets them into trouble.",
            "The character got in trouble a lot at school.",
            "The character has done many things for which they could have been (or were) arrested."
        ],
        [# Peculiarity (5 items; Comm α = .86, Pat α = .82)
            "The character is a strange person.",
            "The character is odd.",
            "The character has been told that their behavior often is bizarre.",
            "The character is considered to be kind of eccentric.",
            "The character would describe themselves as an abnormal person." # Revised due to original being reverse-keyed.
        ],
        [# Perfectionism (6 items; Comm α = .81, Pat α = .85)
            "The character expects nothing less than perfection.",
            "The character only considers a task finished when it's perfect.",
            "The character is unhappy until all the details are taken care of.",
            "The character sets high standards for themselves and others.",
            "The character demands perfection in others.",
            "The character strives in every way possible to be flawless."
        ],
        [# Relationship Insecurity (7 items; Comm α = .84, Pat α = .83)
            "The character is always worried that their partner is going to leave them.",
            "The character is usually convinced that their friends and romantic partners will betray them.",
            "The character gets jealous easily.",
            "The character usually believes that their friends will abandon them.",
            "The character is paralyzed by a fear of rejection.",
            "The character is insecure in their relationships.", # Revised due to original being reverse-keyed.
            "The character generally doubts that their partners to be faithful to them." # Revised due to original being reverse-keyed.
        ],
        [# Rigidity (10 items; Comm α = .77, Pat α = .80)
            "The character dislikes reading or hearing opinions that go against their way of thinking.",
            "The character finds it difficult to consider as valid opinions that differ from their own.",
            "The character has been told that they are rigid and inflexible.",
            "The character has fixed opinions.",
            "The character is often accused of being narrow-minded.",
            "The character is convinced that their way is the best way.",
            "The character believes strongly that the world would be a much better place if they had their way.",
            "The character is inflexible when they think they're right.",
            "The character finds it difficult to compromise in policy debates.",
            "The character believes that most questions have one right answer."
        ],
        [# Risk Taking (5 items; Comm α = .84, Pat α = .84)
            "The character loves dangerous situations.",
            "The character likes to do frightening things.",
            "The character gets a thrill out of doing things that might kill them.",
            "The character would do anything to get an adrenaline rush.",
            "The character prefers risk over safety." # Revised due to original being reverse-keyed.
        ],
        [# Romantic Disinterest (6 items; Comm α = .83, Pat α = .89)
            "The character seldom thinks much about sex.",
            "The character has little desire for sex or romance.",
            "The character could easily live without having sex.",
            "The character seldom enjoys sexual experiences intensely.", # Revised due to original being reverse-keyed.
            "The character sees little need for romance in their life.",
            "The character hates the feeling of being intimately close with someone." # Revised due to original being reverse-keyed.
        ],
        [# Rudeness (7 items; Comm α = .81, Pat α = .80)
            "The character insults people.",
            "The character ridicules people.",
            "The character says inappropriate things.",
            "The character shoots their mouth off.",
            "The character has a mouth that gets them into trouble.",
            "The character has a reputation for asking inappropriate questions.",
            "The character is known for saying offensive things."
        ],
        [# Self Harm (7 items; Comm α = .87, Pat α = .86)
            "The character has urges to cut themselves.",
            "The character has thoughts of injuring themselves.",
            "The character feels that cutting themselves helps them feel better.",
            "The character frequently has thoughts about killing themselves.",
            "The character has written a suicide note before.",
            "The character has intentionally done themselves physical harm.",
            "The character has no will to live."
        ],
        [# Social Withdrawal (6 items; Comm α = .83, Pat α = .87)
            "The character hates going to social gatherings.", # Revised due to original being reverse-keyed.
            "The character feels uncomfortable around people.", # Revised due to original being reverse-keyed.
            "The character keeps to themselves even when they're around other people.",
            "The character rarely enjoys being with people.",
            "The character feels far away from people.",
            "The character finds it difficult to approach others."
        ],
        [# Submissiveness (6 items; Comm α = .81, Pat α = .85)
            "The character is easily controlled by others in their life.",
            "The character lets others take advantage of them.",
            "The character lets themselves be pushed around.",
            "The character prefers that others make the major decisions in their life.",
            "The character lets themselves be directed by others.",
            "The character needs others to help run their life."
        ],
        [# Unusual Beliefs (7 items; Comm α = .83, Pat α = .84)
            "The character believes they have supernatural powers.",
            "The character believes they can see into the future.",
            "The character believes they are able to read the minds of others.",
            "The character believes they have the power to cast spells on others.",
            "The character believes they can control objects with their mind.",
            "The character believes they can use magic to ward off bad thoughts about them.",
            "The character believes they can predict the outcome of events."
        ],
        [# Unusual Experiences (7 items; Comm α = .84, Pat α = .82)
            "The character feels at times that they have left their body and are somehow outside their physical self.",
            "The character sees strange figures or visions when nothing is really there.",
            "The character hears voices talking about them when nobody is really there.",
            "The character has had the feeling that they might not be human.", # Probably need to change this one out with something dynamic if we want fantasy characters e.g. elves.
            "The character has had the feeling that they were someone else.",
            "The character sometimes thinks the TV is talking directly to them.", # Ditto with this one. Hard to feel like the TV is talking to them if they're from an era where the TV hasn't been invented (or ever will).
            "The character feels as if their body, or a part of it, has disappeared."
        ],
        [# Workaholism (6 items; Comm α = .83, Pat α = .85)
            "The character works too much.",
            "The character is a workaholic, with little time for fun or pleasure.",
            "The character has noticed that they put their work ahead of too many other things.",
            "The character works longer hours than most people.",
            "The character works so hard that their relationships have suffered.",
            "The character pushes themselves very hard to succeed."
        ]
    ]

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

    personality_trait_schema

    if not non_axis_traits_only:
        traits = combine_traits(axis_traits)

        selected_traits = random.sample(traits, 1)
        if non_axis_traits:
            selected_traits += random.sample(non_axis_trait_list, 1)

    if non_axis_traits_only:
        selected_traits = random.sample(non_axis_trait_list, 1)
# 33
    if big_five_traits:
        # Although all 5 traits can be put onto a character card, for simplicity's sake, only 1 is selected at this time.
        selected_traits = random.sample(big_five_traits, 1)
        if non_axis_traits:
            logger.warning("Warning: big_five_traits and non_axis_trait_list contain contradictory elements. Combining the two will likely confuse the LLM and produce inconsistent characters.")
            selected_traits += random.sample(non_axis_trait_list, 1)

    if cat_personality_disorder:
        selected_traits = random.sample(cat_personality_disorder, 5)

    # Return the combined string, with each sentence on a new line
    return selected_traits[0]


# Test function to see if I can truncate Llama text.
def truncate(llama: Llama, input: str, maxlen: int) -> str:
    return llama.detokenize(llama.tokenize(input)[:maxlen])


##########################################################
#### MakeDatasetMultiturnConversationSimple Functions ####
##########################################################

def ensure_multiple_answers_are_same(info, conv, initialized_model, override_llm_presets=False):  # why is this a whole separate function? Once upon a time, LLMs were used in validation here, too. But programmatic validation SEEMS to catch the common problems. This is here so that I can add it back in if I have to.
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
        retry = make_multiturn_conversation(info, initialized_model, override_llm_presets=False)

        if retry is not None:  # Note: retry CANNOT actually be None
            c = retry
        else:
            # If we failed to generate a retry, don't waste compute
            logger.info("'ensure_multiple_answers_are_same' failed for this conversation! Returning None")
            return None

    return None

def make_multiturn_conversation(info, initialized_model, override_llm_presets=False):

    conv, conv_output = multi_turn_conversation(
        info[0], info[1], info[2], info[3], 
        initialized_model, 
        override_llm_presets=False, 
        assistant_mode=ASSISTANT_MODE
    )  # based on what was originally: multi_turn_conversation(qa_tuples, character, scenario, scenario_plan, initialized_model)

    write_output_to_file(conv_output, "./multiturn_conversation_generations", info[4])

    return conv

def make_regenerate_answer_constrain_to_text_plan(prompt, qatuple, dissenting_reasoning, initialized_model):
    retries = 0
    prompt_content = {
        "dissenting_reasoning": strip_steps(dissenting_reasoning),
        "qatuple": qatuple,
    }

    # Load the prompt and the grammar.
    try:
        decision_prompt, answer_constrain_to_text_plan_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "answer_constrain_to_text_plan_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

    while retries < 5:

        try:
            start_time = time.time()
            logger.info(f"Generating 'make_regenerate_answer_constrain_to_text_plan' completion...")

            if override_llm_presets:
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

            logger.info(f"\n*** make_regenerate_answer_constrain_to_text_plan COMPLETION ***: \n{completion}\n ***make_regenerate_answer_constrain_to_text_plan COMPLETION ***\n")

            end_time = time.time()
            logger.info(f"Done! Completion took {(end_time - start_time) / 60} minutes to generate.")
            logger.info(f"Completion for 'make_regenerate_answer_constrain_to_text_plan' function generated. Extracting correction...")

            completion_pattern = re.compile(
                r"Reasoning and thought process:\n(.+)", 
                re.DOTALL
            )

            correction = completion_pattern.search(completion).group(1)
            logger.info(f"\n*** make_regenerate_answer_constrain_to_text_plan CORRECTION ***: \n{correction}\n ***make_regenerate_answer_constrain_to_text_plan CORRECTION ***\n")

            return correction

        except Exception as e:
            retries += 1
            logger.exception(f"An Exception occured with completion creation in '{inspect.currentframe().f_code.co_name}' function: {e}\nHere's the completion:\n{completion}")

def multi_turn_conversation(qatuples, character, scenario, scenario_plan, initialized_model, override_llm_presets=False, assistant_mode=False):
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

    conv_starters_filtered = [starter for starter in conv_starters if starter not in first_words_of_card]
    conv_starter = random.choice(conv_starters_filtered)
    logger.info(f"--CONV STARTERS FILTERED--\n{conv_starters_filtered}")

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
            "charname": charname,
            "format_qatuples_qatuples": format_qatuples(qatuples),
        }

        # Load the assistant prompt.
        try:
            cot_prompt, _ = load_external_prompt_and_grammar("multi_turn_conversation_assistant_mode", "answer_constrain_to_text_plan_grammar", prompt_content)
        except Exception as e:
            logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its assistant prompt: {e}")

    else:
        # Load the content to put in the regular prompt.
        extra_info = extract_steps(scenario_plan)
        prompt_content = {
            "charname": charname,
            "scenario": scenario,
            "scenario_plan": scenario_plan,
            "character": character,
            "extra_info": extra_info,
            "conv_starter": conv_starter,
            "format_qatuples_qatuples": format_qatuples(qatuples),
        }

        # Load the regular prompt.
        try:
            cot_prompt, _ = load_external_prompt_and_grammar("multi_turn_conversation", "answer_constrain_to_text_plan_grammar", prompt_content)
        except Exception as e:
            logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt: {e}")

    # NOTE: Very rarely, the first message of this conv will just be part of the character card, causing the conv to not make much sense. 
    # The cause of this is likely the fact that Elise quotes her character card in her first message. 
    # However, referencing the character card in this way also makes characters act as they are described, which is deemed advantageous enough that I am not changing this for now.
    # I get the sense that LLMs can learn relationships and connections between parts of the prompt, even if they're quite far apart, if you give them examples like this. 
    # It's fascinating to see how each part of the prompt has consequences -- sometimes unintended ones.

    # Note: performance degrades rapidly if you put more than one sentence in a pre-prompt parentheses thing
    try:
        start_time = time.time()
        logger.info(f"Generating 'multi_turn_conversation' completion...")

        if override_llm_presets:
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

            logger.info(f"\n*** multi_turn_conversation COMPLETION ***: \n{completion}\n ***multi_turn_conversation COMPLETION ***\n")

    except Exception as e:
        logger.exception(f"An Exception occured in 'multi_turn_conversation' function while generating its completion: {e}")

    # Extract plan
    response_pattern = re.compile(
        f"Conversation that answers the provided question \(be sure that you do not change the questions or answers themselves; {charname} will answer the questions, not ask them; the questions and answers provided should be copied word for word, and surrounded by compelling conversation\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )

    generation = response_pattern.search(completion).group(1)
    logger.info(f"\n*** multi_turn_conversation GENERATION:***\n\n-------------------\n\n {generation} \n*** multi_turn_conversation GENERATION: ***\n\n-------------------\n\n")

    # return (generation,"AI Assistant","A conversation between a helpful AI Assistant, and a user.","N/A",qatuples), completion

    return (generation, character, scenario, scenario_plan, qatuples), completion

###################################################
#### ReturnMultiturnConversationInfo Functions #### TODO: Make the file folders editable.
###################################################

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
        cot_prompt, scenario_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "scenario_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

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

def format_qatuples(qatuples):
    strlst = []
    for qatuple in qatuples:
        strlst.append(
            f"""Question: \"\"\"{qatuple[0]}\"\"\"
Answer: \"\"\"{qatuple[1]}\"\"\""""
        )
    return "\n\n".join(strlst)

def format_qatuples_noquotes(qatuples):
    strlst = []
    for idx, qatuple in enumerate(qatuples):
        strlst.append(f"""{idx + 1}. {qatuple[0]}""")
    return "\n".join(strlst)

# multiturn helpers
# These will probably be used for multiturn rapid-fire answering.

# Idea: use multiple short answers to train the task of answering multiple questions in one response. 
# Two-three short answers per response should be enough.
def make_multiturn_character(qa_tuples, conv_id, initialized_model):
    # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file
    if (ASSISTANT_MODE):  
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
    if (ASSISTANT_MODE):  
        return (qa_tuples, "will", "be", "replaced", conv_id)

    # thought_plan = create_thought_plan_many_tuples(qa_tuples,character,scenario,initialized_model)
    # There IS a way to make multiturn chain of thought answering work: 
    # Namely, generate each pair of messages using a separate prompt or a separate function, each of which has only the thought plan for that question/answer pair. 
    # But simply cramming in all the step-by-step things will confuse the hell out of the poor model. 
    # So for the first release version we're skipping it and just giving the response, with no reasoning, in the multiturn convs.
    character, instructions = make_multiturn_character(qa_tuples, conv_id, initialized_model)
    scenario, scenario_plan = make_multiturn_scenario(qa_tuples, character, conv_id, initialized_model )

    return (qa_tuples, character, scenario, scenario_plan, conv_id)

def make_multiturn_scenario(qa_tuples, character, conv_id, initialized_model):
    max_retries = 3
    attempts = 0

    # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file.
    if (ASSISTANT_MODE):  
        return "will_be_replaced", "will_be_replaced"

    # Create a scenario plan based on a character card and a QA tuple.
    plan, scenario_plan_output = create_scenario_plan_many_tuples(qa_tuples, character, initialized_model)
    write_output_to_file(scenario_plan_output, "./multiturn_scenario_plan_generations", conv_id)

    # Create a scenario based on a character card, a scenario plan, and a QA tuple.
    scenario, scenario_output = create_scenario_many_tuples(qa_tuples, character, plan, initialized_model)  
    write_output_to_file(scenario_output, "./multiturn_scenario_generations", conv_id)

    return scenario, plan

def random_name(NAMES):
    return random.choice(NAMES)

############################################
#### GenerateQATuplesAdvanced Functions ####
############################################

def check_answer(qatuple, initialized_model, permissive_mode=True, override_llm_presets=False):  # The fuck is permissive_mode???
    # Initialize variables
    retries = 0
    prompt_content = {
        "qatuple": qatuple,
    }

    # Load the prompt and the grammar.
    try:
        decision_prompt, answer_accurate_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "answer_accurate_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in '{inspect.currentframe().f_code.co_name}' function while trying to import its prompt and grammar: {e}")

    while retries <= 4:

        # Load the initialized LLM and check the accuracy of the answer in the QA tuple.
        try:
            start_time = time.time()
            logger.info(f"Generating 'check_answer' completion... \nCurrent Retry Count: {retries}")

            # Check if the LLM's settings have been overridden by another Node, then route appropriately.
            if override_llm_presets is True:
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

def check_answer_relevancy_with_text(qatuple, initialized_model, override_llm_presets=False): #TODO: Document this function.
    retries = 0
    prompt_content = {
            "qatuple": qatuple,
    }

    # Load the prompt and the grammar.
    try:
        decision_prompt, answer_relevant_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "answer_relevant_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

    while retries <= 4:

        # Load the initialized LLM and check the QA tuple's answer's relevancy to the question by comparing it to the original text.
        try:
            start_time = time.time()
            logger.info(f"Generating 'check_answer_relevancy_with_text' completion... \nCurrent Retry Count: {retries}")

            if override_llm_presets is True:
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

def check_question(qatuple, initialized_model, override_llm_presets=False): #TODO: Document this function.
    retries = 0
    prompt_content = {
            "qatuple": qatuple,
    }

    # Load the prompt and the grammar.
    try:
        decision_prompt, question_relevant_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "question_relevant_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

    while retries <= 4:

        # Load the initialized LLM and check the question in the QA tuple.
        try:
            start_time = time.time()
            logger.info(f"Generating 'check_question' completion... \nCurrent Retry Count: {retries}")

            if override_llm_presets is True:
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

def generate_new_question(qatuple, initialized_model, override_llm_presets=False):
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    """
    # Determine which paragraphs are worthy of making questions from
    made_questions = False
    retries = 0
    questions = []
    prompt_content = {
        "qatuple": qatuple,
    }

    # Load the prompt and the grammar.
    try:
        question_prompt, question_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name, "question_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

    while not made_questions and (retries <= 5):  # TODO - UPDATE and TEST the few-shot prompt with the latest from generate_questions

        logger.info(f"--QA TUPLE DURING NEW Q GEN--\n{qatuple}")
        start_time = time.time()
        logger.info(f"Generating 'generate_new_question' function completion... \nCurrent Retry Count: {retries}")

        if override_llm_presets is True:
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
        logger.info(f"\nResponse for 'generate_new_question' function ***\n{generation}\n*** Response for 'generate_new_question' function ***")

        print("-------------------")

        pattern = re.compile(
            r"(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)", 
            re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        matches = pattern.findall(generation)
        logger.info(f"\nMatches for 'generate_new_question' function ***\n{matches}\n*** Response for 'generate_new_question' function ***")

        if len(matches) > 0:
            logger.info("Made Qs, yay!")
            made_questions = True
        else:
            logger.info("retry!")
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

def generate_questions(para_tuple, plan, initialized_model, override_llm_presets=False): #TODO: Document this function.
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    :param para_tuple: A tuple consisting of a paragraph of source text and source meta-data.
    :param plan: A question plan created by the generate_questions_plan function.
    :param initialized_model: An initialized LLM (external).
    :param override_llm_presets: A boolean, where True overrides the function's LLM generation presets.
    :return questions:
    :return completion:
    """
    # Determine which paragraphs are worthy of making questions from
    made_questions = False
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

    while not made_questions and (retries <= 5):
        # print("DEBUG\n\n" + decision_prompt)

        start_time = time.time()
        logger.info(f"Generating 'generate_new_question' completion... \nCurrent Retry Count: {retries}")

        if override_llm_presets is True:
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

def generate_questions_plan(para, initialized_model, override_llm_presets=False):
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    """
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
        logger.exception(f"An Exception occurred in {inspect.currentframe().f_code.co_name} function while trying to import its prompt and grammar: {e}")

    start_time = time.time()
    logger.info("Generating 'generate_questions_plan' completion for paragraph...")

    if override_llm_presets is True:
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

def strip_steps(instruction_text):
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

def vet_answer_accuracy_loop(qa_tuple, total_retries, initialized_model, run_id, override_llm_presets=False):
    try:
        qtuple = qa_tuple
        logger.info(f"\n\nStarting ACCURACY loop for question: \nqtuple: {qtuple[0]} \ncontext: {qtuple[2]}\n\n")

        passed_checks = 0
        times_checked = 0
        dissenting_reasoning = ""

        while times_checked < DOUBLE_CHECK_COUNTER: # What the hell is this?
            logger.info(f"\n\nACCURACY CALL CHECK ANSWER: \nqtuple: {qtuple[0]} \ncontext: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}\n\n")

            judgement, answer_accuracy_output = check_answer(qtuple, initialized_model, override_llm_presets)
            write_output_to_file(answer_accuracy_output, "./check_answer_accuracy_generations", run_id)

            if not judgement[0]:  # if not accurate
                dissenting_reasoning = judgement[1]
            else:
                passed_checks += 1

            times_checked += 1
            if passed_checks >= ceil(DOUBLE_CHECK_COUNTER / 2):
                break
            failed_checks = times_checked - passed_checks
            if failed_checks >= ceil(DOUBLE_CHECK_COUNTER / 2):
                break

        if passed_checks >= ceil(DOUBLE_CHECK_COUNTER / 2):  # if question checks passed
            logger.info(f"\n\nANSWER ACCURACY CHECKS PASSED \nretries: {total_retries}\n\n")
            return qtuple
        else:
            # Generate new question and restart the loop
            logger.info(f"\n\nACCURACY CHECKS FAILED - SENDING BACK TO QUESTION LOOP \nretries: {total_retries}\n\n")

            total_retries += 1
            qtuple, generate_new_q_output = generate_new_question(qtuple, initialized_model, override_llm_presets=False)
            write_output_to_file(generate_new_q_output, "./regenerate_question_generations", run_id)

            vet_question_loop(qtuple, initialized_model, total_retries, run_id=run_id.split("--subquestion--")[0], override_llm_presets=False)  # going to get one hell of a call stack by the end of this, but it should be fine

    except Exception as e:
        logger.exception(f"An Exception occurred for vet_answer_accuracy_loop function within generate_qa_tuples function: {e}")
        pass

    return (None, None, None, qtuple[3])

def vet_answer_relevance_loop(qa_tuple, initialized_model, total_retries, run_id, override_llm_presets=False):
    try:
        qtuple = qa_tuple
        logger.info(f"\n\nStarting RELEVANCE loop for question: \nqutple: {qtuple[0]} \ncontext: {qtuple[2]}\n\n")

        passed_checks = 0
        times_checked = 0
        dissenting_reasoning = ""

        while times_checked < DOUBLE_CHECK_COUNTER:
            logger.info(f"\n\nRELEVANCE CALL CHECK ANSWER: \nqtuple: {qtuple[0]} \ncontext: {qtuple[2]} \nretries: {total_retries} \ndissenting reasoning: {dissenting_reasoning}\n\n")

            judgement, answer_relevancy_output = check_answer_relevancy_with_text(qtuple, initialized_model, override_llm_presets)
            write_output_to_file(answer_relevancy_output, "./check_answer_relevancy_generations", run_id)

            if not judgement[0]:  # if not relevant
                dissenting_reasoning = judgement[1]
            else:
                passed_checks += 1
            times_checked += 1

            if passed_checks >= ceil(DOUBLE_CHECK_COUNTER / 2):
                break

            failed_checks = times_checked - passed_checks

            if failed_checks >= ceil(DOUBLE_CHECK_COUNTER / 2):
                break

        if passed_checks >= ceil(DOUBLE_CHECK_COUNTER / 2):
            logger.info(f"\n\nRELEVANCE CHECKS PASSED\n\n")

            return vet_answer_accuracy_loop(qtuple, total_retries, initialized_model, run_id, override_llm_presets=False)
        else:
            logger.info(f"\n\nRELEVANCE CHECKS FAILED - SENDING BACK TO QUESTION LOOP\n\n")

            total_retries += 1
            qtuple, generate_new_q_output = generate_new_question(qtuple, initialized_model)
            write_output_to_file(generate_new_q_output, "./regenerate_question_generations", run_id)

            return vet_question_loop(qtuple, initialized_model, total_retries, run_id=run_id.split("--subquestion--")[0], override_llm_presets=False)

    except Exception as e:
        logger.exception(f"An Exception occurred for vet_answer_relevance_loop function within generate_qa_tuples function: {e}")
        pass

    return (None, None, None, qtuple[3])

def vet_question_loop(qa_tuple, initialized_model, total_retries, run_id=None, override_llm_presets=False):
    try:
        question_group_id = run_id # Hacky...
        qtuple = qa_tuple
        logger.info(f"\n\nStarting QUESTION loop for question: n\qtuple:{qtuple[0]}, context: {qtuple[2]}\n\n")

        while total_retries <= 4:
            run_id = question_group_id + "--subquestion--" + make_id()
            passed_checks = 0
            times_checked = 0
            dissenting_reasoning = ""

            while times_checked < DOUBLE_CHECK_COUNTER:
                logger.info(f"\n\nQUESTION CALL CHECK ANSWER: \n{qtuple[0]}\ncontext: {qtuple[2]}\nretries: {total_retries} \ndissenting reasoning: {dissenting_reasoning}\n\n")

                judgement, check_q_output = check_question(qtuple, initialized_model, override_llm_presets=False)
                write_output_to_file(check_q_output, "./check_question_generations", run_id)

                if not judgement[0]:  # if not relevant
                    dissenting_reasoning = judgement[1]
                else:
                    passed_checks += 1

                times_checked += 1
                if passed_checks >= ceil(DOUBLE_CHECK_COUNTER / 2):
                    break
                failed_checks = times_checked - passed_checks
                if failed_checks >= ceil(DOUBLE_CHECK_COUNTER / 2):
                    break

            if passed_checks >= ceil(DOUBLE_CHECK_COUNTER / 2):  # if all question checks passed
                logger.info(f"\n\nQUESTION CHECKS PASSED retries: {total_retries}")
                return vet_answer_relevance_loop(qtuple, initialized_model, total_retries, run_id, override_llm_presets=False)
            else:
                # Generate new question and restart the loop
                logger.info(f"\n\nQUESTION CHECKS FAILED - GENERATING NEW QUESTION \nretries: {total_retries}\n\n")
                total_retries += 1

                if (total_retries <= 4):  # only regen question if we're not already at max regens
                    qtuple, generate_new_q_output = generate_new_question(qtuple, initialized_model)
                    write_output_to_file(generate_new_q_output,"./regenerate_question_generations",run_id,)
                    logger.info("New question: ", qtuple)
                # no calling of vet_question_loop, since we're already in a while loop

    except Exception as e:
        logger.exception(f"An Exception occurred in vet_question_loop function within generate_qa_tuples function: {e}")

    return (None, None, None, qtuple[3])

######################
#### NODE CLASSES ####
######################

#TODO Replace absolute references to prompts and grammars with relative ones. Also add try-excepts to them.

class GenerateQATuplesSimple: # TODO Actually make this class. But I need to move the function innards of GenerateQATuplesAdvanced outside of it to do that.
    """
    This function takes a 

    :param filtered_worthy_for_questions: Output from the JudgeParagraphs function
    :param initialized_model: An initialized model, with parameters set for logic and determinism.
    :param qa_tuple_directory_name: The name of directory where the QA tuples will go.
    :param total_retries: The total number of retries the function should run.
    :return: List of sentence chunks with source text information
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filtered_worthy_for_questions": ("TUPLE", {"forceInput": True}),
                "initialized_model": ("INITIALIZED_MODEL",),
                "qa_tuple_directory_name": ("STRING", {"default": 'qatuples_raw'}),
            },
        }
    RETURN_TYPES = ("OUTPUT_TEXT",)
    RETURN_NAMES = ("vetted_qa_tuples",)

    FUNCTION = "generate_qa_tuples"

    CATEGORY = "output_generation"

class GenerateQATuplesAdvanced: #TODO Write function documentation. This class will be a BEAR to create, and will likely need to be split up into separate nodes later on depending on its complexity.
    """
    This node generates QA tuples from input paragraphs.
    :param filtered_worthy_for_questions: Filtered Output from the JudgeParagraphs node. Default format: tuples of a paragraph chunk and source meta-data.
    :param initialized_model: An initialized model.
    :param qa_tuple_directory_name: The name of directory where the QA tuples will go.
    :param total_retries: The total number of retries the QA tuples should be cross-checked before being saved to output.
    :return vetted_qa_tuples: A list of QA tuples, vetted for relevance and accuracy.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "initialized_model": ("INITIALIZED_MODEL",),
                "qa_tuple_directory_name": ("STRING", {"default": 'qatuples_raw'}),
                "question_plan_directory_name": ("STRING", {"default": 'question_plan_generations'}),
                "question_plan_generations_directory_name": ("STRING", {"default": 'question_generation_generations'}),
                "total_retries": ("INT", {"default": 3, "min": 0, "max": 10000, "step":1}), # TODO: Maybe add dynamic total_retries option, so that it cuts off after a threshold of good-to-bad tuples is reached?
            },
            "optional": {
                "filtered_worthy_for_questions": ("TUPLE", {"forceInput": True}),
                "override_llm_presets": ("OVERRIDE_LLM_PRESETS_CHOICE",),
            },
        }
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("vetted_qa_tuples",)

    FUNCTION = "generate_qa_tuples"

    CATEGORY = "output_generation"

    def generate_qa_tuples(self, initialized_model, qa_tuple_directory_name, question_plan_directory_name, question_plan_generations_directory_name, total_retries, filtered_worthy_for_questions=None, override_llm_presets=False):

        node_start_time = time.time()
        # Set directory for QA tuples, and make it if it doesn't exist.
        qa_tuples_dir = f"./output/{qa_tuple_directory_name}"
        if not os.path.exists(qa_tuples_dir):
            os.makedirs(qa_tuples_dir)

        # Initialize vetted_qa_tuples
        vetted_qa_tuples = []  # tuple list of QA tuples that have been judged good

        # Attempt to initialize filtered_worthy_for_questions. If NameError occurs, create an empty array for the filtered_worthy_for_questions variable
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
            # For every paragraph and their index number in the judged paragraphs...
            for idx, para in enumerate(tqdm(filtered_worthy_for_questions)):
                # for idx, para in enumerate(tqdm(filtered_worthy_for_questions[:10])): # Use this instead if you are just testing all steps of the notebook
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
                    plan, questions_plan_output = generate_questions_plan(para, initialized_model, override_llm_presets=False) ###### WORKS
                    write_output_to_file(questions_plan_output, f"./output/{question_plan_directory_name}", question_group_id)

                    # Create QAs from the input paragraph and question plan, then write it to a file.
                    logger.info(f"\n\n\nOUTER LOOP CALL GENERATE Q: \npara: {para}, \n\n idx: {idx} \n\n plan: {plan}"        )
                    question_answer_tuples, question_generation_output = generate_questions(para, plan, initialized_model, override_llm_presets=False) ######
                    write_output_to_file(question_generation_output, f"./output/{question_plan_generations_directory_name}", question_group_id)

                    # Begin vetting the QA.
                    for qnum, question_answer_tuple in enumerate(question_answer_tuples):
                        logger.info(f"\n\n=======!!=BEGIN VETTING QA TUPLE {idx}_{qnum}=!!=======\n\n")
                        good_qa_tuple = vet_question_loop(question_answer_tuple, initialized_model, total_retries, run_id=question_group_id, override_llm_presets=False) ######

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

        # Create basic summary stats for the outputs so far, including None values.
        logger.info("-------------- QUESTIONS CREATED ------------- STATS SO FAR (may be wrong if run was continued from interruption):")
        nones = list(filter(lambda x: x[0] is None, vetted_qa_tuples))
        logger.info(f"\nNones: {len(nones)} \nNon-nones: {len(vetted_qa_tuples) - len(nones)} \nTotal: {len(vetted_qa_tuples)}")
        time.sleep(3)

        # filter out all None values
        vetted_qa_tuples = [qa for qa in vetted_qa_tuples if qa[0] is not None]
        node_end_time = time.time()
        logger.info(f"\nGenerateQATuplesAdvanced node complete! \nTotal Runtime: {(node_end_time - node_start_time) / 60} minutes.")

        return(vetted_qa_tuples,)

#TODO Write function documentation.
#TODO Add in LLM preset override functionality.
#TODO Investigate function and class set-up. Something seems wrong here...
class ReviseQATuples:
    """
    This function revises a QA tuple. It's really a fuck-ton of functions in a trench-coat.

    :param filtered_worthy_for_questions: Output from the JudgeParagraphs function
    :param initialized_model: An initialized model, with parameters set for logic and determinism.
    :param qa_tuple_directory_name: The name of directory where the AQ tuples will go.
    :param total_retries: The total number of retries the function should run.
    :return vetted_qa_tuples: List of sentence chunks with source text information
    """

    @staticmethod
    def extract_question_answer(response):
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
    def check_qatuple_context(qatuple, initialized_model):
        retries = 0
        while retries <= 4:

            # Load the QA tuple into the prompt_content dictionary so it can be put into the prompt.
            prompt_content = {
                "qatuple": qatuple,
            }

            # Load in the prompt and grammar.
            decision_prompt = format_external_text_like_f_string(PROMPT_DICT['check_qatuple_context'], prompt_content)
            check_qatuple_context_grammar = LlamaGrammar.from_string(GRAMMAR_DICT['check_qatuple_context_grammar'])

            try: # Check the QA tuple's context using the LLM.
                start_time = time.time()
                logger.info("Generating 'check_qatuple_context' completion for qatuple...")

                completion = initialized_model(
                    decision_prompt,
                    max_tokens=10000,
                    stop=["</s>", "# Input:"],
                    echo=True,
                    grammar=check_qatuple_context_grammar,
                    temperature=0.2,
                )["choices"][0]["text"]

                end_time = time.time()
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
                "initialized_model": ("INITIALIZED_MODEL",),
                "qatuple_directory_name": ("STRING", {"default": 'qatuples_revised'}),
            },
        }
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("vetted_qa_tuples",)
    FUNCTION = "revise_qa_tuples"

    CATEGORY = "output_generation"

    def revise_qa_tuples(self, vetted_qa_tuples, initialized_model, qatuple_directory_name):
        # Check for and fix the common mistake: mentioning "the text".
        # TODO refactor to be continuable, should take like 30 mins at most

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

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            logger.info(f"Loading file: {file_path}")

                            if content == "failed":
                                vetted_qa_tuples.append(None)
                            else:
                                try:
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
                            logger.error(f"JSON decode error with the contents in revise_qa_tuples function in class ReviseQATuples: {content}")
                            # Handle the error appropriately

                try:
                    revision_id = make_id()
                    revision, revision_output = self.check_qatuple_context(tup, initialized_model)
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


        # Print stats related to revised qatuples, and filter out nones (questions that were unanswerable due to lack of context).

        logger.info("-------------- QUESTIONS REVISED ------------- STATS SO FAR:")
        nones = list(filter(lambda x: x is None, vetted_qa_tuples))
        logger.info(f"\nNones: {len(nones)}\nNon-nones: {len(vetted_qa_tuples) - len(nones)}\nTotal: {len(vetted_qa_tuples)}")

        # filter out all None values
        vetted_qa_tuples = [qa for qa in vetted_qa_tuples if qa is not None]
        logger.info("---------------- ONTO EXAMPLES GENERATION-------------------")

        return(vetted_qa_tuples,)

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
                    if helper_functions.has_sequential_chars(question,dict_q,N_CHARACTERS_SAME):
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

    CATEGORY = "output_generation"

    def group_vetted_qa_tuple_by_paragraph(self, vetted_qa_tuples, check_for_matching_subtrings_anywhere):

        if check_for_matching_subtrings_anywhere == "True":
            qa_tuples_by_paragraph = self.group_by_text(vetted_qa_tuples, check_for_matching_subtrings_anywhere=True)
        else:
            qa_tuples_by_paragraph = self.group_by_text(vetted_qa_tuples)

        return(qa_tuples_by_paragraph,)

class ReturnMultiturnConversationInfo: # TODO Write function documentation. Oh, and finish the function. And I don't care what anyone else says, this WILL become multiple nodes so help me god!
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
                "initialized_model": ("INITIALIZED_MODEL",),
                "multi_turn_convs_info_dir": ("STRING", {"default": 'multi_turn_convs_info_dir'},),
                "assistant_mode_arg": (["Off","On"],),
                "arrangements_to_take": ("INT", {"default:": 3, "min": 1, "max": 10, "step":1},),
                "purge_loaded_llm_from_memory_after_node_is_done": (["Off","On"],),
            },
        }
    RETURN_TYPES = ("OUTPUT_TEXT","PURGE_TRIGGER",)
    RETURN_NAMES = ("multi_turn_convs_info",)
    FUNCTION = "return_multi_turn_conversation_info"

    CATEGORY = "output_generation"

    def return_multi_turn_conversation_info(self, qa_tuples_by_paragraph, initialized_model, multi_turn_convs_info_dir, assistant_mode_arg, arrangements_to_take, purge_loaded_llm_from_memory_after_node_is_done):
        # Set the assistant_mode boolean variable.
        if assistant_mode_arg == "On":
            assistant_mode = True
        else:
            assistant_mode = False

        if not os.path.exists(f"./{multi_turn_convs_info_dir}"):
            os.makedirs(f"./{multi_turn_convs_info_dir}")

        multi_turn_convs_info = []

        logger.info(f"\n*** INPUT FOR qa_tuples_by_paragraph *** \n{qa_tuples_by_paragraph} \n*** INPUT FOR qa_tuples_by_paragraph *** ")

        for idx, group in enumerate(qa_tuples_by_paragraph):
            logger.info(f"\n*** Current qa_tuples_by_paragraph *** \ngroup:{group} \nidx:{idx} ")
            all_permutations = list(itertools.permutations(group))
            logger.info(f"all_permutations:\n{all_permutations}\nall_permutations")

            sample_size = min(arrangements_to_take, len(all_permutations))
            logger.info(f"sample_size:\n{sample_size}\sample_size")

            sampled_permutations = random.sample(all_permutations, sample_size)
            logger.info(f"sampled_permutations:\n{sampled_permutations}\sampled_permutations")

            group_convs_info = []

            for iter, perm in enumerate(sampled_permutations):
                file_path = os.path.join(multi_turn_convs_info_dir, f"info_{idx}_{iter}.json")

                # Skip if file already exists
                if not os.path.exists(file_path):
                    info = make_multiturn_conversation_info(perm, initialized_model)

                    if info is not None:
                        with open(file_path, "w") as file:
                            json.dump(info, file, indent=4)

                    group_convs_info.append(info)
                else:
                    logger.info(f"Skipped generating {file_path} as it already exists")

            multi_turn_convs_info.append(group_convs_info)

        # Set-off the purge trigger. Note that this does NOT automatically purge the loaded LLM from RAM and/or VRAM. 
        # Rather, it produces an boolean that tells the PurgeLlmFromRamOrVram node whether to do that or not.
        if purge_loaded_llm_from_memory_after_node_is_done == "On":
            purge_trigger = True
        else:
            purge_trigger = False

        return (multi_turn_convs_info, purge_trigger,)

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
                "initialized_model": ("INITIALIZED_MODEL",),
                "multi_turn_convs_info_dir": ("STRING", {"default": 'multi_turn_convs_info_dir'},),
                "multi_turn_convs_dir_arg" : ("STRING", {"default": 'multi_turn_convs_dir'},),
                "read_from_external_json": (["True","False"],),
                "use_default_output_dir": (["False","True"],),
            },
            "optional": {
                "multi_turn_convs_info": ("OUTPUT_TEXT",)
            },
            "hidden": {},
        }
    RETURN_TYPES = ()
    FUNCTION = "make_dataset_multi_turn_conversation"

    OUTPUT_NODE = True

    CATEGORY = "output_generation"

    def make_dataset_multi_turn_conversation(self, initialized_model, multi_turn_convs_info_dir, multi_turn_convs_dir_arg, read_from_external_json, use_default_output_dir, multi_turn_convs_info=None):

        # Set up the multi-turn conversation and results list.
        multi_turn_convs = []
        results = list()

        # Hardcode this as False for the moment.
        # TODO Actually implement overrides. This will allow for testing purposes.
        override_llm_presets=False

        # Set up the output directory
        if use_default_output_dir == "False":
            multi_turn_convs_dir = f"./{multi_turn_convs_dir_arg}"
            # Make the output directory if it doesn't exist.
            if not os.path.exists(multi_turn_convs_dir):
                os.makedirs(multi_turn_convs_dir)
        else:
            multi_turn_convs_dir = self.output_dir

        # Option to load in the conversation info directly from the previous function instead of reading it from json files.
        # Some people generating datasets have access to serious compute. Why slow them down?
        if read_from_external_json == "True": 
            convs_info = self.read_json_files_info(multi_turn_convs_info_dir) 
        elif read_from_external_json == "False" and multi_turn_convs_info is not None:
            convs_info = multi_turn_convs_info
        else:
            logger.error(f"Could not load data into 'make_dataset_multi_turn_conversation' function in class MakeDatasetMultiturnConversationSimple.")
            print("This was likely caused because 'read_from_external_json' is False and 'multi_turn_convs_info' is not connected to an input node.")

        try:
            # For all the information and their indexes within the conversation information...
            for idx, info in enumerate(convs_info):
                #Set the file path for the multi-turn conversation output.
                file_path = os.path.join(multi_turn_convs_dir, f"conv_{idx}.json")

                # Skip if the file already exists
                if not os.path.exists(file_path):
                    # Make a multi-turn conversation out of the information.
                    conv = make_multiturn_conversation(info, initialized_model)

                    # Make sure the multiple answers in the conversation are the same.
                    final_conv = ensure_multiple_answers_are_same(info, conv, initialized_model)

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

        return { "ui": { "results": results } }

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

    CATEGORY = "helper_function"

    def convert_directory_to_list(directory_path_arg):
        # Set up lists and paths.
        directory_path = f"./{directory_path_arg}/"
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

# This is NOT a terminal node, as the outputs of it could be useful for diagnostic purposes.
class ConvertDirectoryAndProcessConversations: #TODO Maybe merge into class ConvertDirectoryToList as a separate function?
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path_arg": ("STRING", {"default": 'multi_turn_convs'}),
            }
        }
    RETURN_TYPES = ("MASTER_OUTPUT_TEXT",)
    FUNCTION = "convert_directory_to_list"

    CATEGORY = "helper_function"

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

        elif master_output_text is not None:
            data = master_output_text

        else:
            logging.error("No data was input into filter_and_flatten function in class FilterAndFlatten.")
            print("Either read it from a JSON or connect it to an input node such as 'Create a Simplified List Copy of the Dataset'.")

        flat_list = []

        # Loop through each sublist in the main list
        for sublst in lst:
            # Check if the first element of the sublist is itself a list (subsublist1)
            if isinstance(sublst[0], list):
                # Extend the flat_list with the elements from subsublist1
                flat_list.extend(sublst[0])

        return(flat_list, { "ui": { "text": len(flat_list) } })

###############################################
#### NODE CLASS MAPPINGS AND NAME MAPPINGS ####
###############################################

NODE_CLASS_MAPPINGS = {
    #Output Generation
    #"GenerateQATuplesSimple": GenerateQATuplesSimple,
    #"CreateCharacterCardPlanManyTuples": CreateCharacterCardPlanManyTuples,
    #"CreateScenarioPlanManyTuples": CreateScenarioPlanManyTuples,
    #"MakeMultiturnCharacter": MakeMultiturnCharacter,
    #"MakeMuliturnScenario": MakeMuliturnScenario,
    "ReturnMultiturnConversationInfo": ReturnMultiturnConversationInfo,
    "ConvertDirectoryAndProcessConversations": ConvertDirectoryAndProcessConversations,
    "ConvertDirectoryToList": ConvertDirectoryToList,
    "MakeDatasetMultiturnConversationSimple": MakeDatasetMultiturnConversationSimple,
    #"MakeMultiturnConversation": MakeMultiturnConversation,
    "FilterAndFlatten": FilterAndFlatten, # DONE, mostly
    "GenerateQATuplesAdvanced": GenerateQATuplesAdvanced, # DONE, mostly
    #"GenerateQATuplesSimple": GenerateQATuplesSimple,
    "GroupVettedTuplesByText": GroupVettedTuplesByText,
    "ReviseQATuples": ReviseQATuples,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    #Output Generation
    #"CreateCharacterCardPlanManyTuples": "Create Character Card Plan (Many Tuples)",
    #"CreateScenarioPlanManyTuples": "Create Scenario Plan (Many Tuples)",
    #"MakeMultiturnCharacter": "Make Multi-turn Character",
    #"MakeMuliturnScenario": "Make Multi-turn Scenario",
    "ConvertDirectoryAndProcessConversations": "Create a Simplified List Copy of the Dataset, then Process the Conversations",
    "ConvertDirectoryToList": "Create a Simplified List Copy of the Dataset",
    "MakeDatasetMultiturnConversationSimple": "Make Dataset: Multi-turn Conversation (Simple)",
    "ReturnMultiturnConversationInfo": "Return Multi-turn Conversation Info",
    #"MakeMultiturnConversation": "Make Dataset (Multi-turn Conversation)",
    "FilterAndFlatten": "Return How Many Lines of Dialogue Were Generated",
    "GenerateQATuplesAdvanced": "Generate QA Tuples (Advanced)",
    #"GenerateQATuplesSimple": "Generate QA Tuples",
    "GroupVettedTuplesByText": "Group Revised QA Tuples by Paragraph",
    "ReviseQATuples": "Revise QA Tuples",
}
