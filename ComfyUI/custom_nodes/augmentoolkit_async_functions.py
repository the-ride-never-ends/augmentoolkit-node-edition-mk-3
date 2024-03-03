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

from augmentoolkit import (
    ASSISTANT_MODE, # Global variables
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
    extract_name, # Functions
    format_external_text_like_f_string,
    format_qatuples,
    load_external_prompt_and_grammar,
    override_prompt_and_grammar,
    strip_steps,
    write_output_to_file,
    special_instructions
)

#########################################
######## ASYNC HELPER FUNCTIONS #########
#########################################

def make_id():
    return str(uuid.uuid4())

async def submit(LLM, prompt, sampling_params):  # Submit request and wait for it to stream back fully
    # Initialized variables
    request_id = make_id()
    engine_wrapper = LLM['llm']
    outputs = []
    final_output = None
 
    async for request_output in engine_wrapper.engine.generate(prompt, sampling_params, request_id):
        outputs.append(request_output.outputs[0].text)
        final_output = request_output
        full_output = "".join(outputs)

        return final_output.prompt + final_output.outputs[0].text

#################################################
######## JudgeParagraphs Async Functions ########
#################################################

async def judge_paragraph(p, LLM):
    # Initialize variables.
    reached_decision = False
    max_retries = 0
    engine_wrapper = LLM['llm']
    prompt_content = {
        "p":p
    }
    
    logger.info(f"\nParagraph being judged: \n{p} \nParagraph being judged \ntype: {type(p)}")
    
    # Load the prompt and grammar.
    try:
        decision_prompt, _ = load_external_prompt_and_grammar("judge_paragraph","dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occured in async 'judge_paragraph' function while trying to load its prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_judge_paragraph_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_aphrodite_sampling_presets') is True:
            logger.info("Overriding default LLM presets for async 'judge_paragraph' function.")
            engine_wrapper = overrides['llm']
            LLM['override_aphrodite_sampling_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            try:
                logger.info("Overriding the prompt for async 'judge_paragraph' function.")
                decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in async 'judge_paragraph' function: {e}")
                print(f"Check the prompt folder. The prompt must be a txt file named 'judge_paragraph', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    except KeyError:
        logger.info("Overrides for async 'judge_paragraph' function not present. Using default presets.")


    while not reached_decision and (max_retries <= 3):

        # Determine whether to override the functions generation presets
        if LLM['override_aphrodite_sampling_presets'] is not None:
            sampling_params = LLM['override_aphrodite_sampling_presets']
        else:
            sampling_params = SamplingParams(
                max_tokens=6000,
                min_p=0.4,
                stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
            )

        # Load the initialized LLM and judge the paragraph.
        try: 
            start_time = time.time()
            logger.info(f"Generating async 'judge_paragraph' completion... \nCurrent Retry Count: {max_retries}")

            completion = await engine_wrapper.submit(
                decision_prompt,
                sampling_params
            )

            end_time = time.time()
            logger.info(f"Completion took {(end_time - start_time) / 60} minutes to async generate.")
            logger.info(f"Completion for async 'judge_paragraph' function on retry {max_retries} generated. Extracting response pattern...")

        except Exception as e:
            logger.exception(f"An Exception occured in async 'judge_paragraph' function while trying to load its prompt: {e}")
            break

        response_pattern = re.compile(
            r"Reasoning and thought process \(reason intelligently\):(.+)",
            re.DOTALL | re.IGNORECASE,
        )

        judgement_pattern = re.compile(
            r"Final Judgment:(.+)", re.DOTALL | re.IGNORECASE
        )

        # Extract the response pattern and determination from the completion.
        try:
            response = response_pattern.search(completion).group(1)
            # print(response)
            determination = judgement_pattern.search(response).group(1)
            # print("\n\nDETERMINATION:\n------")
            # print(determination)
            # print("\n---------\n")
            if "unsuitable" in determination.lower():
                reached_decision = True
                return (None, p[1])
            elif "suitable" in determination.lower():
                return (p[0], p[1])
        except:
            logger.exception(f"An exception occured in async 'judge_paragraph' function under class JudgeParagraphs: {e}")
            break

        max_retries += 1

## Paragraph Filtering (worthy for questions?)
async def determine_worthy(idx, p, judged_worthy_for_questions, LLM, output_dir,):

    # for idx, p in tqdm(enumerate(paragraphs_processed[:10])):
    file_name = f"{idx}.json"
    file_path = os.path.join(output_dir, file_name)

    # Check if the judgement for this paragraph already exists
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
            print("LOADING: ", data)
        if isinstance(data, str):
            judged_worthy_for_questions.append((None, data[7:]))
        else:
            judged_worthy_for_questions.append((data["paragraph"], data["metadata"]))

    else: #Judge the paragraph for QA sutability.
        judgement = await judge_paragraph(p, LLM)
        judged_worthy_for_questions.append(judgement)

        # Prepare the data to be written to the file
        if judgement[0] is not None:
            # The paragraph passed the judgement
            data_to_write = {"paragraph": judgement[0], "metadata": judgement[1]}
        else:
            # The paragraph did not pass the judgement
            data_to_write = f"failed|{judgement[1]}"

        # Write the judgement to a unique file as JSON
        with open(file_path, "w") as file:
            json.dump(data_to_write, file)

        # Debug messages
        try:
            if judgement[0] is not None:
                print(f"DEBUG model decided that index {idx} was suitable")
            else:
                print(f"DEBUG model decided that index {idx} was not suitable")
        except:
            print(f"DEBUG max retries exceeded for index {idx}")

# Scheduler function for the judge paragraph node functions.
async def filter_all_questions(paragraphs_processed, judged_worthy_for_questions, engine_wrapper, output_dir,take_subset=False,):
    # Schedule everything.
    if not take_subset:
        tasks = [
            determine_worthy(
                idx, # Index number
                p, # Paragraph
                judged_worthy_for_questions, # List to put worth paragraphs in.
                engine_wrapper, # LLM
                output_dir, #O utput directory 
            )
            for idx, p in enumerate(paragraphs_processed)
        ]
    else:
        tasks = [
            determine_worthy(
                idx,
                p,
                judged_worthy_for_questions,
                engine_wrapper,
                output_dir,
            )
            for idx, p in enumerate(paragraphs_processed[:13]) #???
        ]

    # More async bullshit!
    for future in tqdmasyncio.tqdm.as_completed(tasks):
        await future

##################################################
######## GenerateQATuples Async Functions ########
##################################################

# Answer vetting
async def check_answer(qatuple, LLM, permissive_mode=True):
    retries = 0
    engine_wrapper = LLM['llm']
    prompt_content = {
        "qatuple": qatuple
    }

    # Load the prompt and grammar.
    try:
        decision_prompt, _ = load_external_prompt_and_grammar("check_answer","dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occured in async 'judge_paragraph' function while trying to load its prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_check_answer_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_aphrodite_sampling_presets') is True:
            logger.info("Overriding default LLM presets for async 'check_answer' function.")
            LLM['override_aphrodite_sampling_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            try:
                logger.info("Overriding the prompt for async 'check_answer' function.")
                decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in async 'check_answer' function: {e}")
                print(f"Check the prompt folder. The prompt must be a txt file named 'check_answer', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    except KeyError:
        logger.info("Overrides for async 'check_answer' function not present. Using default presets.")
        
    while retries <= 4:
        try:
            # Determine whether to override the functions generation presets
            if LLM['override_aphrodite_sampling_presets'] is not None:
                sampling_params = LLM['sampling_params']
            else:
                sampling_params = SamplingParams(
                    max_tokens=6000,
                    stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
                    temperature=0.2,
                )

            completion = await engine_wrapper.submit(decision_prompt, sampling_params)

            completion_pattern = re.compile(
                r"Reasoning and thought process \(the text is your single source of truth\):\n(.+)",
                re.DOTALL,
            )
            response = completion_pattern.search(completion).group(1).strip()
            if DEBUG_MODE:
                logger.info(f"\n\RESPONSE:\n------\n{response}\n---------\n")

            if permissive_mode:
                determination_pattern = re.compile(
                    r"Overall Accuracy Determination:(.+)", re.DOTALL
                )
                determination = determination_pattern.search(response).group(1).strip()
            else:
                determination = response
            if DEBUG_MODE:
                logger.info(f"\n\nDETERMINATION:\n------\n{determination}\n---------\n")

            if (
                "inaccurate" in determination.lower()
                or "Inaccurate" in determination.lower()
                or "mostly" in determination.lower()
                or "partial" in determination.lower()
            ):  # The "mostly" is there to catch "mostly accurate" which the model says occasionally, and which actually means inaccurate.
                return (False, response), completion
            elif (
                "accurate" in determination or "Accurate" in determination
            ):  # very deliberate placement of accurate here, becaues the model can sometimes say irrelevant at the very end, even after saying accurate in its judgement
                return (True, response), completion
            elif (
                "irrelevant" in determination or "Irrelevant" in determination
            ):  # optional support for checking relevance here, too.
                return (
                    None,
                    response,
                ), completion  # signal that question is irrelevant
            else:
                Exception("Broke!")

        except Exception as e:
            retries += 1
            logger.exception(f"An Exception occured in async 'check_answer' function: {e}")
            traceback.print_exc()

# Answer vetting
# For now, this checks answer relevancy too. The danger with abstracting answer relevancy into a separate step is that anything which relies on knowledge that is obviously mentioned in the text already up until this point, will get screwed
async def check_answer_relevancy_with_text(qatuple, LLM):
    retries = 0
    engine_wrapper = LLM['llm']
    prompt_content = {
        "qatuple": qatuple
    }

    # Load the prompt and grammar.
    try:
        decision_prompt, _ = load_external_prompt_and_grammar("check_answer_relevancy_with_text","dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occured in async 'check_answer_relevancy_with_text' function while trying to load its prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_check_answer_relevancy_with_text_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_aphrodite_sampling_presets') is True:
            logger.info("Overriding default LLM presets for async 'check_answer_relevancy_with_text' function.")
            LLM['override_aphrodite_sampling_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            try:
                logger.info("Overriding the prompt for async 'check_answer_relevancy_with_text' function.")
                decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in async 'check_answer_relevancy_with_text' function: {e}")
                print(f"Check the prompt folder. The prompt must be a txt file named 'check_answer_relevancy_with_text', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    except KeyError:
        logger.info("Overrides for async 'check_answer_relevancy_with_text' function not present. Using default presets.")

    while retries <= 4:
        
        try:
            if LLM['override_aphrodite_sampling_presets'] is True:
                sampling_params = overrides['sampling_params']
            else:
                sampling_params = SamplingParams(
                    max_tokens=5500,
                    stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
                    temperature=0.2,
                )

            completion = await engine_wrapper.submit(decision_prompt, sampling_params)
            completion_pattern = re.compile(
                r"Reasoning and thought process \(be careful about extra details, even vague ones\):\n(.+)",
                re.DOTALL | re.IGNORECASE,
            )
            judgement_pattern = re.compile(
                r"Explanation of Judgment:(.+)", re.DOTALL | re.IGNORECASE
            )
            response = completion_pattern.search(completion).group(1).strip()
            # # print(response)
            determination = judgement_pattern.search(response).group(1).strip()

            if DEBUG_MODE:
                logger.info(f"\n\nDETERMINATION:\n------\n{determination}\n---------\n")

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
            # print(
            #     f"Something went catastrophically wrong with this one. Investigate! Here's the completion:\n{completion}"
            # )
            traceback.print_exc()
    return None, None

# A separate prompt for the reword step of checking qatuple context, since the grammar is bugged on the original
async def check_qatuple_context(qatuple, LLM):
    retries = 0
    engine_wrapper = LLM['llm']
    prompt_content = {
        "qatuple": qatuple
    }

    # Load the prompt.
    try:
        decision_prompt, _ = load_external_prompt_and_grammar("check_qatuple_context","dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occured in async 'check_qatuple_context' function while trying to load its prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If it doesn't, use the function's defaults.
    try: 
        overrides = LLM['override_check_qatuple_context_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_aphrodite_sampling_presets') is True:
            logger.info("Overriding default LLM presets for async 'check_qatuple_context' function.")
            LLM['override_aphrodite_sampling_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            try:
                logger.info("Overriding the prompt for async 'check_qatuple_context' function.")
                decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in async 'check_qatuple_context' function: {e}")
                print(f"Check the prompt folder. The prompt must be a txt file named 'check_qatuple_context', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    except KeyError:
        logger.info("Overrides for async 'check_qatuple_context' function not present. Using default presets.")

    while retries <= 4:

        if LLM['override_aphrodite_sampling_presets'] is True:
            sampling_params = overrides['sampling_params']
        else:
            sampling_params = SamplingParams(
                max_tokens=10000,
                stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
                temperature=0.2,
            )

        try:
            start_time = time.time()
            logger.info("Generating async 'check_qatuple_context' completion for qatuple...")

            completion = await engine_wrapper.submit(decision_prompt, sampling_params)

            end_time = time.time()
            if DEBUG_MODE:
                logger.info(f"\n***Completion for async 'check_qatuple_context' function ***\n{completion}\n*** Completion for async 'generate_questions_plan' function ***")
            logger.info(f"Completion took {(end_time - start_time) / 60} minutes to generate.")
            logger.info(f"Completion for async 'check_qatuple_context' function for retry {retries} generated. Extracting response pattern...")

            response_pattern = re.compile(
                r"Reasoning and thought process \(be thorough\):(.+)",
                re.DOTALL | re.IGNORECASE,
            )
            response = response_pattern.search(completion).group(1).strip()

            decision_pattern = re.compile(r"Final judgment:(.+)", re.IGNORECASE)
            if DEBUG_MODE:
                logger.info(f"\n*** Response for async 'check_qatuple_context' function ***\n{response}\n*** Response for 'check_qatuple_context' function ***")
            
            determination = decision_pattern.search(response).group(1).strip()
            logger.info(f"\n\nDETERMINATION:\n------\n{determination}\n---------\n")


            if "pass" in determination.lower():
                logger.info("Leaving be...")
                return (True, response), completion
            elif "reword" in determination.lower():
                logger.info("Rewording...")
                q, a = extract_question_answer(response)
                logger.info((q, a, qatuple[2], qatuple[3]))
                return (q, a, qatuple[2], qatuple[3]), completion
            elif "fail" in determination.lower():
                logger.info("Setting to None...")
                return (False, response), completion
            else:
                logger.info("Did not contain relevant or irrelevant! Retrying")
                retries += 1

        except Exception as e:
            logger.exception(f"An Exception occured in async 'check_qatuple_context' function: {e}")
            if retries <= 4:
                retries += 1
            else:
                return (None, None), None

    return (None, None), None

# Answer vetting
async def check_question(qatuple, LLM):
    retries = 0
    engine_wrapper = LLM['llm']
    prompt_content = {
        "qatuple": qatuple
    }

    # Load the prompt.
    try:
        decision_prompt, _ = load_external_prompt_and_grammar("check_question","dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occured in async 'check_question' function while trying to load its prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If it doesn't, use the function's defaults.
    try: 
        overrides = LLM['override_check_question_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_aphrodite_sampling_presets') is True:
            logger.info("Overriding default LLM presets for async 'check_question' function.")
            LLM['override_aphrodite_sampling_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            try:
                logger.info("Overriding the prompt for async 'check_question' function.")
                decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in async 'check_question' function: {e}")
                print(f"Check the prompt folder. The prompt must be a txt file named 'check_question', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    except KeyError:
        logger.info("Overrides for async 'check_question' function not present. Using default presets.")

    while retries <= 4:
        try:
            # Determine whether to override the functions generation presets
            if LLM['override_aphrodite_sampling_presets'] is not None:
                sampling_params = LLM['sampling_params']
            else:
                sampling_params = SamplingParams(
                    max_tokens=4000,
                    stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
                    temperature=0.2,
                )

            completion = await engine_wrapper.submit(decision_prompt, sampling_params)

            response_pattern = re.compile(
                r"Reasoning and thought process \(be careful around \"how\" and \"why\" questions\):(.+)",
                re.DOTALL | re.IGNORECASE,
            )
            response = response_pattern.search(completion).group(1).strip()
            if DEBUG_MODE:
                logger.info(f"\n\nRESPONSE:\n------\n{response}\n---------\n")

            decision_pattern = re.compile(
                r"Final Judgment:(.+)", re.DOTALL | re.IGNORECASE
            )
            determination = decision_pattern.search(response).group(1).strip()
            if DEBUG_MODE:
                logger.info(f"\n\nDETERMINATION:\n------\n{determination}\n---------\n")

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
                print("Did not contain relevant or irrelevant! Retrying")
                retries += 1
        except Exception as e:
            print("Exception!", e)
            traceback.print_exc()
            if retries <= 4:
                retries += 1
            else:
                return (None, None), completion
    return (None, None), None

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
        print("Returned none, failed to match")
        return None, None

async def generate_new_question(qatuple, LLM):
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    """
    # Determine which paragraphs are worthy of making questions from
    made_questions = False
    retries = 0
    engine_wrapper = LLM['llm']
    questions = []
    prompt_content ={
        "qatuple": qatuple
    }

    # Load the prompt.
    try:
        question_prompt, _ = load_external_prompt_and_grammar("generate_new_question","dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occured in async 'generate_new_question' function while trying to load its prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If it doesn't, use the function's defaults.
    try: 
        overrides = LLM['override_generate_new_question_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_aphrodite_sampling_presets') is True:
            logger.info("Overriding default LLM presets for async 'generate_new_question' function.")
            LLM['override_aphrodite_sampling_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            try:
                logger.info("Overriding the prompt for async 'generate_new_question' function.")
                question_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in async 'generate_new_question' function: {e}")
                print(f"Check the prompt folder. The prompt must be a txt file named 'generate_new_question', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    except KeyError:
        logger.info("Overrides for async 'generate_new_question' function not present. Using default presets.")

    # TODO - UPDATE and TEST the few-shot prompt with the latest from generate_questions
    while not made_questions and (retries <= 5):  
        try:    
            # Determine whether to override the functions generation presets
            if LLM['override_aphrodite_sampling_presets'] is not None:
                sampling_params = LLM['sampling_params']
            else:
                sampling_params = SamplingParams(
                    max_tokens=8000,
                    stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
                    temperature=0.2,
                )
            logger.info(f"--QA TUPLE DURING NEW Q GEN--\n{qatuple}\n")

            completion = await engine_wrapper.submit(question_prompt, sampling_params)
            if DEBUG_MODE:
                logger.info(f"COMPLETION:\n\n----------------------\n{completion}\n------------------")

            # Extract questions
            response_pattern = re.compile(
                r"Question \(based on text\):\n(.+)", re.IGNORECASE | re.DOTALL
            )
            generation = response_pattern.search(completion).group(1)
            if DEBUG_MODE:
                logger.info(f"GENERATION:\n\n-------------------\n{generation}\n-------------------")
                            
            pattern = re.compile(
                r"(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)",
                re.DOTALL | re.MULTILINE | re.IGNORECASE,
            )
            matches = pattern.findall(generation)

            if len(matches) > 0:
                print("Made Qs, yay!")
                made_questions = True
            else:
                print("retry!")
                retries += 1

        except Exception as e:
            logger.exception(f"An Exception occured in async 'generate_new_question' function: {e}")
            break

        for match in matches:
            return (
                match[0].replace(") ", "", 1).strip(),
                match[1].replace(") ", "", 1).strip(),
                qatuple[2].replace(") ", "", 1),
                qatuple[3],
            ), completion
        logger.warning(f"Should not have reached here\nmatches: {matches}\nquestions:{questions}")

    return questions, completion

# Question generation
async def generate_qatuples_from_para(idx, para, LLM=None, vetted_qa_tuples=None, qa_tuples_dir=None, double_check_counter=3,):

    try: # check if the questions file already exist
        existing_files = glob.glob(
            os.path.join(qa_tuples_dir, f"para_{idx}_*.json")
        ) 

        if len(existing_files) > 0:  # If files exist, skip this paragraph entirely
            logger.info(f"Skipping para_{idx} as files already exist; loading said files")
            for file_path in existing_files:
                with open(file_path, "r") as file:
                    qa_tuple = tuple(json.load(file))
                vetted_qa_tuples.append(qa_tuple)
                continue

        question_group_id = make_id()
        if DEBUG_MODE:
            logger.info(f"\n\n\nOUTER LOOP CALL GENERATE QPLAN para: {para}, \n\n idx: {idx}")

        (plan, questions_plan_output,) = await generate_questions_plan(para, LLM)
        
        write_output_to_file(questions_plan_output, "./question_plan_generations", question_group_id)
        if DEBUG_MODE:
            logger.info(f"\n\n\nOUTER LOOP CALL GENERATE Q: {para}, \n\n idx: {idx} \n\n plan: {plan}")

        (question_answer_tuples, question_generation_output,) = await generate_questions(para, plan, LLM)
        
        write_output_to_file(question_generation_output,"./question_generation_generations", question_group_id,)
        for qnum, question_answer_tuple in enumerate(question_answer_tuples):
            logger.info(f"\n\n=======!!=BEGIN VETTING QA TUPLE {idx}_{qnum}=!!=======\n\n")
            good_qa_tuple = await vet_question_loop(
                question_answer_tuple,
                0,
                question_group_id=question_group_id,
                LLM=LLM,
                double_check_counter=double_check_counter,
            )

            # Write resulting question file if the tuple is not None
            if good_qa_tuple[0] is not None:
                file_path = os.path.join(qa_tuples_dir, f"para_{idx}_q_{qnum}.json")
                with open(file_path, "w") as file:
                    json.dump(good_qa_tuple, file, indent=4)

            vetted_qa_tuples.append(good_qa_tuple) 
            # We must filter out all None values at the end; 
            # but appending Nones lets us know where things went wrong, and how often.

    except Exception as e:
        logger.exception(f"Q ERROR: {e}")
        traceback.print_exc()

async def generate_questions(para_tuple, plan, LLM):
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    """
    # Determine which paragraphs are worthy of making questions from
    made_questions = False
    retries = 0
    questions = []
    engine_wrapper = LLM['llm']
    prompt_content = {
        "para_tuple": para_tuple,
        "strip_steps_plan": strip_steps(plan)
    }
    
    # Load the prompt and grammar.
    try:
        question_prompt, _ = load_external_prompt_and_grammar("generate_questions","dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occured in async 'judge_paragraph' function while trying to load its prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_generate_questions_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_aphrodite_sampling_presets') is True:
            logger.info("Overriding default LLM presets for async 'generate_questions' function.")
            LLM['override_aphrodite_sampling_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            try:
                logger.info("Overriding the prompt for async 'generate_questions' function.")
                decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in async 'generate_questions' function: {e}")
                print(f"Check the prompt folder. The prompt must be a txt file named 'generate_questions', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    except KeyError:
        logger.info("Overrides for async 'check_answer' function not present. Using default presets.")

    while not made_questions and (retries <= 5):
        try:
            # Determine whether to override the functions generation presets
            if LLM['override_aphrodite_sampling_presets'] is not None:
                sampling_params = LLM['sampling_params']
            else:
                sampling_params = SamplingParams(
                    max_tokens=12000,
                    stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
                    temperature=0.8,
                    top_k=-1,
                    top_p=1,
                    min_p=0.5,
                )
            completion = await engine_wrapper.submit(question_prompt, sampling_params)
    
            # Extract questions
            response_pattern = re.compile(
                r"Questions \(make 4\):\n(.+)", re.IGNORECASE | re.DOTALL
            )
            generation = response_pattern.search(completion).group(1)
            if DEBUG_MODE:
                logger.info(f"GENERATION:\n\n-------------------\n\n{generation}\n")

            pattern = re.compile(
                r"(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)",
                re.DOTALL | re.MULTILINE | re.IGNORECASE,
            )
            matches = pattern.findall(generation)

            if len(matches) > 0:
                made_questions = True
            else:
                retries += 1

        except Exception as e:
            logger.exception(f"An Exception occured in 'generate_questions' function: {e}")
            break

        if retries > 5:
            return None, None

    for match in matches:
        questions.append(
            (
                match[0].replace(") ", "", 1).strip(),
                match[1].replace(") ", "", 1).strip(),
                para_tuple[0].replace(") ", "", 1),
                para_tuple[1].replace(") ", "", 1),
            )
        )

    return questions, completion

async def generate_questions_plan(text, LLM):
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    """
    engine_wrapper = LLM['llm']
    prompt_content = {
        "text": text,
    }
    
    # Load the prompt and grammar.
    # Determine which paragraphs are worthy of making questions from
    # Analyze-Realize-Create-Example loop
    try:
        cot_prompt, _ = load_external_prompt_and_grammar("generate_questions_plan","dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occured in async 'generate_questions_plan' function while trying to load its prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_generate_questions_plan_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_aphrodite_sampling_presets') is True:
            logger.info("Overriding default LLM presets for async 'generate_questions_plan' function.")
            LLM['override_aphrodite_sampling_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            try:
                logger.info("Overriding the prompt for async 'generate_questions_plan' function.")
                decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in async 'generate_questions_plan' function: {e}")
                print(f"Check the prompt folder. The prompt must be a txt file named 'generate_questions_plan', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    except KeyError:
        logger.info("Overrides for async 'check_answer' function not present. Using default presets.")

    try:
        # Determine whether to override the functions generation presets
        if LLM['override_aphrodite_sampling_presets'] is not None:
            sampling_params = LLM['sampling_params']
        else:
            sampling_params = SamplingParams(
                max_tokens=8000,
                stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
                temperature=0.8,
                top_k=-1,
                top_p=1,
                min_p=0.5,
            )
        completion = await engine_wrapper.submit(cot_prompt, sampling_params)

        # Extract plan
        response_pattern = re.compile(
            r"Reasoning and thought process \(being careful to only plan questions that are entirely based on the text provided\):\n(.+)",
            re.IGNORECASE | re.DOTALL,
        )
        generation = response_pattern.search(completion).group(1)
        if DEBUG_MODE:
            logger.info(f"GENERATION:\n\n-------------------\n\n{generation}")

    except Exception as e:
        logger.exception(f"An Exception occured in 'generate_questions' function: {e}")

    return generation, completion

# Postprocessing function for question/answer validation
async def repair_qatuple_context(idx, tup, LLM, writepath, vetted_qa_tuples):

    file_path = os.path.join(writepath, f"revised_{idx}.json")

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()  # Read the file once and store its content
            logger.info(file_path)
            if content == "failed":
                print("Loaded failed file")
                vetted_qa_tuples[idx] = None
                return None
            logger.info(f"Loaded file:\n{content}")

            try:
                data = json.loads(content)  # Convert the string back to JSON
                vetted_qa_tuples[idx] = (data[0], data[1], data[2], data[3])
                return None
            except json.JSONDecodeError:
                logger.exception("JSON decode error with the contents:", content)

    try:
        revision_id = make_id()
        revision, revision_output = await check_qatuple_context(tup, LLM)

        # incidentally, identifying the problem and fixing it in the same step (without another planning step) 
        # works a lot better than identifying it and then trying to fix it in the next step.
        write_output_to_file(revision_output, "./question_context_revision_generations", revision_id)  

        if isinstance(revision[0], str):  # if the thing was reworded
            vetted_qa_tuples[idx] = revision
        elif not revision[0]:
            vetted_qa_tuples[idx] = None  
        # prepare item for deletion later; right now we just store it as None because indexes
        # else, if it passed, we just leave it be.

        # Write in-progress
        if not os.path.exists(writepath):
            os.makedirs(writepath)

        if vetted_qa_tuples[idx]:
            with open(file_path, "w") as file:
                json.dump(vetted_qa_tuples[idx], file, indent=4)
        else:
            with open(file_path, "w") as file:
                file.write("failed")

    except Exception as e:
        logger.exception(f"!!! ERROR!\n{e}")
        traceback.print_exc()

# Control flow helpers -- Question/Answer Validation
async def vet_answer_accuracy_loop(qa_tuple, total_retries, run_id, LLM=None, double_check_counter=3):
    try:
        qtuple = qa_tuple
        if DEBUG_MODE:
            logger.info(f"\n\nStarting ACCURACY loop for question: {qtuple[0]}, context: {qtuple[2]}")

        passed_checks = 0
        times_checked = 0
        dissenting_reasoning = ""
        while times_checked < double_check_counter:
            if DEBUG_MODE:
                logger.info(f"\n\nACCURACY CALL CHECK ANSWER: {qtuple[0]}, context: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}")

            judgement, answer_accuracy_output = await check_answer(qtuple, LLM)
            write_output_to_file(answer_accuracy_output, "./check_answer_accuracy_generations", run_id)

            if not judgement[0]:  # if not accurate
                dissenting_reasoning = judgement[1]
            else:
                passed_checks += 1
            times_checked += 1
            if passed_checks >= ceil(double_check_counter / 2):
                break
            failed_checks = times_checked - passed_checks
            if failed_checks >= ceil(double_check_counter / 2):
                break

        if passed_checks >= ceil(double_check_counter / 2):  # if question checks passed
            if DEBUG_MODE:
                logger.info(f"\n\ANSWER ACCURACY CHECKS PASSED retries: {total_retries}")

            return qtuple
        else:
            # Generate new question and restart the loop
            if DEBUG_MODE:
                logger.info(f"\n\nACCURACY CHECKS FAILED - SENDING BACK TO QUESTION LOOP retries: {total_retries}")
            total_retries += 1

            (qtuple, generate_new_q_output,) = await generate_new_question(qtuple, LLM)
            write_output_to_file(generate_new_q_output, "./regenerate_question_generations", run_id)

            vet_question_loop(
                qtuple,
                total_retries,
                question_group_id=run_id.split("--subquestion--")[0],
                LLM=LLM,
                double_check_counter=double_check_counter,
            )  # going to get one hell of a call stack by the end of this, but it should be fine

    except Exception as e:
        logger.info(f"!!ERROR!!\n{e}")
        traceback.print_exc()

    return (None, None, None, qtuple[3])

async def vet_answer_relevance_loop(qa_tuple, total_retries, run_id, LLM=None, double_check_counter=3):
    try:
        qtuple = qa_tuple
        if DEBUG_MODE:
            logger.info(f"\n\nStarting RELEVANCE loop for question: {qtuple[0]}, context: {qtuple[2]}")

        passed_checks = 0
        times_checked = 0
        dissenting_reasoning = ""
        while times_checked < double_check_counter:
            if DEBUG_MODE:
                logger.info(f"\n\nRELEVANCE CALL CHECK ANSWER: {qtuple[0]}, context: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}")

            (judgement, answer_relevancy_output,) = await check_answer_relevancy_with_text(qtuple, LLM)
            
            write_output_to_file(answer_relevancy_output, "./check_answer_relevancy_generations", run_id)

            if not judgement[0]:  # if not relevant
                dissenting_reasoning = judgement[1]
            else:
                passed_checks += 1
            times_checked += 1
            if passed_checks >= ceil(double_check_counter / 2):
                break
            failed_checks = times_checked - passed_checks
            if failed_checks >= ceil(double_check_counter / 2):
                break

        if passed_checks >= ceil(double_check_counter / 2):
            logger.info(f"\n\nRELEVANCE CHECKS PASSED")

            return await vet_answer_accuracy_loop(
                qtuple,
                total_retries,
                run_id,
                LLM=LLM,
                double_check_counter=double_check_counter,
            )
        else:
            logger.info(f"\n\nRELEVANCE CHECKS FAILED - SENDING BACK TO QUESTION LOOP")
            total_retries += 1

            (qtuple, generate_new_q_output,) = await generate_new_question(qtuple, LLM)
            write_output_to_file(generate_new_q_output, "./regenerate_question_generations", run_id)
            
            return vet_question_loop(
                qtuple,
                total_retries,
                question_group_id=run_id.split("--subquestion--")[0],
                LLM=LLM,
                double_check_counter=double_check_counter,
            )

    except Exception as e:
        logger.info(f"!!ERROR!!\n{e}")
        traceback.print_exc()

    return (None, None, None, qtuple[3])

async def vet_question_loop(qa_tuple, total_retries, question_group_id=None, LLM=None, double_check_counter=3,):
    try:
        qtuple = qa_tuple
        if DEBUG_MODE:
            logger.info(f"\n\nStarting QUESTION loop for question: {qtuple[0]}, context: {qtuple[2]}")

        while total_retries <= 4:
            run_id = question_group_id + "--subquestion--" + make_id()
            passed_checks = 0
            times_checked = 0
            dissenting_reasoning = ""

            while times_checked < double_check_counter:
                if DEBUG_MODE:
                    logger.info(f"\n\nQUESTION CALL CHECK ANSWER: {qtuple[0]}, context: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}")

                judgement, check_q_output = await check_question(qtuple, LLM)
                write_output_to_file(check_q_output, "./check_question_generations", run_id)

                if not judgement[0]:  # if not relevant
                    dissenting_reasoning = judgement[1]
                else:
                    passed_checks += 1
                times_checked += 1
                if passed_checks >= ceil(double_check_counter / 2):
                    break
                failed_checks = times_checked - passed_checks
                if failed_checks >= ceil(double_check_counter / 2):
                    break

            if passed_checks >= ceil(double_check_counter / 2): # if all question checks passed
                if DEBUG_MODE:
                    logger.info(f"\n\nQUESTION CHECKS PASSED retries: {total_retries}")

                return await vet_answer_relevance_loop(
                    qtuple,
                    total_retries,
                    run_id,
                    LLM=LLM,
                    double_check_counter=double_check_counter,
                )
            else:
                # Generate new question and restart the loop
                if DEBUG_MODE:
                    logger.info(f"\n\nQUESTION CHECKS FAILED - GENERATING NEW QUESTION retries: {total_retries}")
                total_retries += 1

                if (total_retries <= 4):  # only regen question if we're not already at max regens
                    (qtuple, generate_new_q_output,) = await generate_new_question(qtuple, LLM)
                    write_output_to_file(generate_new_q_output,"./regenerate_question_generations",run_id,)
                    logger.info(f"New question: {qtuple}")
                # no calling of vet_question_loop, since we're already in a while loop

    except Exception as e:
        logger.exception(f"!!ERROR!!\n{e}")
        traceback.print_exc()

    return (None, None, None, qtuple[3])


#########################################################
#### ReturnMultiturnConversationInfo Async Functions ####
#########################################################

# Idea: use multiple short answers to train the task of answering multiple questions in one response. Two-three short answers per response should be enough.
async def make_multiturn_character(qa_tuples, conv_id, LLM, assistant_mode):
    if (assistant_mode):  # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file
        return "will_be_replaced", "will_be_replaced"

    (plan, instructions, card_plan_output,) = await create_character_card_plan_many_tuples(qa_tuples, LLM)  # I will reuse the many tuples function for short question-answers, there's a lot of prompting in here already
    write_output_to_file(card_plan_output, "./multiturn_card_plan_generations", conv_id)

    (char, char_output,) = await create_character_card_many_tuples(qa_tuples, plan, instructions, LLM)  # creates a character card
    write_output_to_file(char_output, "./multiturn_card_generations", conv_id)

    return char, instructions

async def make_multiturn_scenario(qa_tuples, character, conv_id, LLM, assistant_mode):
    if (assistant_mode):  # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file
        return "will_be_replaced", "will_be_replaced"

    (plan, scenario_plan_output,) = await create_scenario_plan_many_tuples(qa_tuples, character, LLM)

    write_output_to_file(scenario_plan_output, "./multiturn_scenario_plan_generations", conv_id)

    # creates a scenario based on a character card and question/answer tuple
    (scenario, scenario_output,) = await create_scenario_many_tuples(qa_tuples, character, plan, LLM)  
    write_output_to_file(scenario_output, "./multiturn_scenario_generations", conv_id)

    return scenario, plan

async def create_scenario_plan_many_tuples(qatuples, character, LLM):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """
    engine_wrapper = LLM['llm']
    # removing the source text makes this much better. 
    # Perfection is achieved not when there's nothing more to add, but when there's nothing left to take away.

    charname = extract_name(character)

    prompt_content = {
        "format_qatuples_qatuples": format_qatuples(qatuples),
        "character": character
    }

    # Load the prompt and grammar.
    try:
        cot_prompt, _ = load_external_prompt_and_grammar("create_scenario_plan_many_tuples","dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occured in async 'create_scenario_plan_many_tuples' function while trying to load its prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_create_scenario_plan_many_tuples_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_aphrodite_sampling_presets') is True:
            logger.info("Overriding default LLM presets for async 'create_scenario_plan_many_tuples' function.")
            LLM['override_aphrodite_sampling_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            try:
                logger.info("Overriding the prompt for async 'create_scenario_plan_many_tuples' function.")
                decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in async 'create_scenario_plan_many_tuples' function: {e}")
                print(f"Check the prompt folder. The prompt must be a txt file named 'create_scenario_plan_many_tuples', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    except KeyError:
        logger.info("Overrides for async 'create_scenario_plan_many_tuples' function not present. Using default presets.")

    try:
        # Determine whether to override the functions generation presets
        if LLM['override_aphrodite_sampling_presets'] is not None:
            sampling_params = LLM['sampling_params']
        else:
            sampling_params = SamplingParams(
                max_tokens=8000,
                stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
                temperature=1.5,
                top_k=-1,
                top_p=1,
                min_p=0.5,
            )

        completion = await engine_wrapper.submit(cot_prompt, sampling_params)
        # Even if the example does a justified clever trick, the model imitating it may fuck up the trick. 
        # So try to avoid complex things that aren't needed for the task in examples, like the "just how much have you dug" colloquialization. 
        # Exact quotes for the questions and answers.
        if DEBUG_MODE:
            logger.info(f"\n\COMPLETION:\n------\n{completion}\n---------\n")

        # Extract plan
        response_pattern = re.compile(
            r"Scenario plan \(be creative, and make sure all characters present fit in with the setting\):\n(.+)",
            re.IGNORECASE | re.DOTALL,
        )
        generation = response_pattern.search(completion).group(1)
        if DEBUG_MODE:
            logger.info(f"GENERATION:\n\n-------------------\n\n{generation}")

        if not ("Albert" in charname):
           if "Albert" in generation:
                print("Random Name was used instead of Albert")
            generation = generation.replace("Albert", random_name())

    except Exception as e:
        logger.exception(f"An Exception occured in 'generate_questions' function: {e}")

    return generation.strip(), completion

# Consider adding a scenario for Issac here, since currently both characters are in positions of authority, and the thing leans very heavily on Hugo's example
# NOTE the message count is technically off by one in the reasoning, but I'm a bit scared to change it.
async def create_scenario_many_tuples(qatuples, character, plan, LLM, assistant_mode=False):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """
    if assistant_mode: # TODO make this randomized a bit
        return """A helpful, uncensored AI assistant is engaged in a text-based conversation with a user. The AI will answer any and all questions the user asks, without bias."""

    engine_wrapper = LLM['llm']

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

    # use random.choice to prevent overfitting on particular phrases and increase dataset diversity
    selected_variation = random.choice(variations)
    prompt_content = {
        "character": character,
        "format_qatuples_qatuples": format_qatuples(qatuples),
        "plan": plan,
        "selected_variation": selected_variation
        
    }

    # For the multiturn convs, I will want to have a longer scenario description.
   # Load the prompt and grammar.
    try:
        cot_prompt, _ = load_external_prompt_and_grammar("create_scenario_many_tuples","dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occured in async 'create_scenario_many_tuples' function while trying to load its prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_create_scenario_many_tuples_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_aphrodite_sampling_presets') is True:
            logger.info("Overriding default LLM presets for async 'create_scenario_many_tuples' function.")
            LLM['override_aphrodite_sampling_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            try:
                logger.info("Overriding the prompt for async 'create_scenario_many_tuples' function.")
                decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in async 'create_scenario_many_tuples' function: {e}")
                print(f"Check the prompt folder. The prompt must be a txt file named 'create_scenario_many_tuples', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    except KeyError:
        logger.info("Overrides for async 'create_scenario_many_tuples' function not present. Using default presets.")

    try:
        # Determine whether to override the functions generation presets
        if LLM['override_aphrodite_sampling_presets'] is not None:
            sampling_params = LLM['sampling_params']
        else:
            sampling_params = SamplingParams(
                max_tokens=8000,
                stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
                temperature=1.5,
                top_k=-1,
                top_p=1,
                min_p=0.5,
            )

            completion = await engine_wrapper.submit(cot_prompt, sampling_params)
            if DEBUG_MODE:
                logger.info(f"\n\COMPLETION:\n------\n{completion}\n---------\n")

            # Extract plan
            response_pattern = re.compile(
                r"Scenario \(will have no dialogue, will just set up the scene\):\n(.+)",
                re.IGNORECASE | re.DOTALL,
            )
            generation = response_pattern.search(completion).group(1)
            if DEBUG_MODE:
                logger.info(f"\n\GENERATION:\n------\n{completion}\n---------\n")

            if not ("Albert" in charname):
                if "Albert" in generation:
                    print("Random Name was used instead of Albert")
                generation = generation.replace("Albert", random_name())

    except Exception as e:
        logger.exception(f"An Exception occured in 'generate_questions' function: {e}")

    return generation, completion

async def make_multiturn_conversation_info(qa_tuples, LLM, assistant_mode):
    conv_id = make_id()

  # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file.
    if (assistant_mode):
        return (qa_tuples, "will", "be", "replaced", conv_id)

    # # There IS a way to make multiturn chain of thought answering work: 
    # generate each pair of messages using a separate prompt or a separate function, 
    # each of which has only the thought plan for that question/answer pair. 
    # But simply cramming in all the step-by-step things will confuse the hell out of the poor model. 
    # So for the first release version we're skipping it and just giving the response, with no reasoning, in the multiturn convs.
    character, instructions = await make_multiturn_character(qa_tuples, conv_id, LLM, assistant_mode)

    scenario, scenario_plan = await make_multiturn_scenario(qa_tuples, character, conv_id, LLM, assistant_mode)

    return (qa_tuples, character, scenario, scenario_plan, conv_id)

def extract_steps(text, steps=[2, 4, 5]):
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

def extract_first_words(character_name, text):
    # Regular expression pattern to extract first word after the character's name
    pattern = rf"{character_name}: \"(\w+)"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    return matches

def extract_author_name(title):
    pattern = re.compile(r"\b(?:by|By)\s+([^,]+),")
    match = re.search(pattern, title)
    if match:
        author_name = match.group(1)
    else:
        author_name = [False]
    return author_name[0]  # first letter of Author name

def select_random_capital(exclusions):
    # Create a list of capital letters excluding the ones in the exclusions list
    capitals = [letter for letter in string.ascii_uppercase if letter not in exclusions]

    # Select a random capital letter from the filtered list
    if capitals:
        return random.choice(capitals)
    else:
        return "No available capital letters to choose from"

def extract_capital_letters(input_string):
    capital_letters = []
    for char in input_string:
        if char.isupper():
            capital_letters.append(char)
    return capital_letters

async def create_character_card_many_tuples(qatuples, plan, instructions, LLM, cheap_mode=False):  # Use cheap mode if you don't have the compute power to crank up the context to 8k using RoPE
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

    engine_wrapper = LLM['llm']
    prompt_content = {
        "qatuples": qatuples,
        "instructions": instructions,
        "plan": plan,
        "starting_str": starting_str,
    }

    # Load the prompt and grammar.
    try:
        cot_prompt, _ = load_external_prompt_and_grammar("create_character_card_many_tuples", "dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in 'create_character_card_many_tuples' function while trying to import its prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_create_character_card_many_tuples_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_aphrodite_sampling_presets') is True:
            logger.info("Overriding default LLM presets for async 'create_character_card_many_tuples' function.")
            LLM['override_aphrodite_sampling_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            try:
                logger.info("Overriding the prompt for async 'create_character_card_many_tuples' function.")
                decision_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in async 'create_character_card_many_tuples' function: {e}")
                print(f"Check the prompt folder. The prompt must be a txt file named 'create_character_card_many_tuples', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")

    except KeyError:
        logger.info("Overrides for async 'create_character_card_many_tuples' function not present. Using default presets.")

    try:
        start_time = time.time()
        # Determine whether to override the functions generation presets
        if LLM['override_aphrodite_sampling_presets']:
            sampling_params = overrides['sampling_params']
        else:
            sampling_params = SamplingParams(
                max_tokens=10000,
                stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
                temperature=2,
                top_k=-1,
                top_p=1,
                min_p=0.5,
            )

        completion = await engine_wrapper.submit(cot_prompt, sampling_params)

        end_time = time.time()
        logger.info(f"Completion took {(end_time - start_time) / 60} minutes to generate.")
        logger.info(f"Completion for 'create_character_card_plan_many_tuples' function generated. Extracting response pattern...")
        if DEBUG_MODE:
            logger.info(f"COMPLETION:\n\n----------------------\n{completion}\n------------------")

        # Extract plan
        response_pattern = re.compile(
            r"Character card \(be creative, write at least 3 paragraphs for each dialogue line\):\n(.+)",
            re.IGNORECASE | re.DOTALL,
        )

        generation = response_pattern.search(completion).group(1)
        if DEBUG_MODE:
            logger.info(f"GENERATION:\n\n-------------------\n\n{generation}\n")

    except Exception as e:
        logger.exception(f"An Exception occured in 'create_character_card_many_tuples' function: {e}")

    return generation, completion

async def create_character_card_plan_many_tuples(qatuples, LLM):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """

    # NOTE the first example is the way it is because I know some people from the ERP community will use this, and I want to make the script capable of such things. Also it might help to jailbreak it a bit. I even considered making it a nonhuman character (ie, catgirl) but I have no idea how to write those.
    # I am not very experienced at writing stuff like the first example, please do not make a dispositional attribution about my personality or writing talent based on some of the cringeworthy stuff there.

    instructions_string = special_instructions(n=1)
    engine_wrapper = LLM['llm']
    prompt_content = {
        "qatuples": qatuples,
        "format_qatuples_qatuples": format_qatuples(qatuples),
        "instructions_string_strip": instructions_string.strip(),
        
    }

    # Load the assistant prompt.
    try:
        cot_prompt, _ = load_external_prompt_and_grammar("create_character_card_plan_many_tuples", "dummy_grammar", prompt_content)
    except Exception as e:
        logger.exception(f"An Exception occurred in async 'create_character_card_plan_many_tuples' function while trying to import its assistant prompt: {e}")

    # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
    try: 
        overrides = LLM['override_create_character_card_plan_many_tuples_presets']

        # Override the default function presets if it's requested.
        if overrides.get('override_llm_presets') is True:
            logger.info("Overriding default LLM presets for async 'create_character_card_plan_many_tuples' function.")
            initialized_model = overrides['llm']
            LLM['override_llm_presets'] = True

        # Override the prompt if it's requested.
        if overrides.get('prompt'):
            question_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
            logger.info("Overriding the prompt for async 'create_character_card_plan_many_tuples' function.")

    except KeyError:
        logger.info("Overrides for async 'create_character_card_plan_many_tuples' function not present. Using default presets.")

    if LLM['override_llm_presets']:
        sampling_params = overrides['sampling_params']
    else:
        sampling_params = SamplingParams(
            max_tokens=8000,
            stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
            temperature=2,
            top_k=-1,
            top_p=1,
            min_p=0.4,
        )

    completion = await engine_wrapper.submit(cot_prompt, sampling_params)
    if DEBUG_MODE:
        logger.info(f"\n\COMPLETION:\n------\n{completion}\n---------\n")

    # Extract plan
    response_pattern = re.compile(
        r"Character card plan \(be creative, do not use real people as characters, do NOT make the author of the book a character\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    generation = response_pattern.search(completion).group(1)
    if DEBUG_MODE:
        logger.info(f"\n\GENERATION:\n------\n{completion}\n---------\n")

    return generation, instructions_string, completion

#########################################################

# Group tuples for multiturn example generation (by chunk of source text) and then run that helper (so that we can make multiturn conversations from questions based on the same paragraphs)
def group_by_text(tuples_list):
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
    return [
        identify_duplicates.identify_duplicates(group)
        for group in list(groups.values())
    ]

# Graphing code generated by GPT-4. May be suboptimal/ugly.
def filter_and_graph(tuples):
    # Count the occurrences of None and non-None for each source text
    source_counts = Counter()
    for paragraph, source in tuples:
        if paragraph is None:
            source_counts[source] = source_counts.get(source, [0, 0])
            source_counts[source][0] += 1
        else:
            source_counts[source] = source_counts.get(source, [0, 0])
            source_counts[source][1] += 1

    # Prepare data for the graph
    labels = list(source_counts.keys())
    none_counts = [source_counts[source][0] for source in labels]
    non_none_counts = [source_counts[source][1] for source in labels]

    # Plotting the graph
    x = range(len(labels))
    plt.bar(x, none_counts, width=0.4, label="Not suitable", align="center")
    plt.bar(x, non_none_counts, width=0.4, label="Valid Paragraphs", align="edge")
    plt.xlabel("Source Text")
    plt.ylabel("Number of Paragraphs")
    plt.title("Paragraphs Suitable for Questions by Source Text")
    plt.xticks(x, labels, rotation="vertical")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Filter out tuples with None and return the new list
    filtered_list = [t for t in tuples if t[0] is not None]
    return filtered_list

## Paragraph Filtering (worthy for questions?)
async def determine_worthy(idx, p, judged_worthy_for_questions, LLM, output_dir,):
    # for idx, p in tqdm(enumerate(paragraphs_processed[:10])):
    file_name = f"{idx}.json"
    file_path = os.path.join(output_dir, file_name)
    # Check if the judgement for this paragraph already exists
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
            print("LOADING: ", data)
        if isinstance(data, str):
            judged_worthy_for_questions.append((None, data[7:]))
        else:
            judged_worthy_for_questions.append((data["paragraph"], data["metadata"]))
    else:
        judgement = await judge_paragraph(p, LLM)
        judged_worthy_for_questions.append(judgement)

        # Prepare the data to be written to the file
        if judgement[0] is not None:
            # The paragraph passed the judgement
            data_to_write = {"paragraph": judgement[0], "metadata": judgement[1]}
        else:
            # The paragraph did not pass the judgement
            data_to_write = f"failed|{judgement[1]}"

        # Write the judgement to a unique file as JSON
        with open(file_path, "w") as file:
            json.dump(data_to_write, file)

        # Debug messages
        try:
            if judgement[0] is not None:
                print(f"DEBUG model decided that index {idx} was suitable")
            else:
                print(f"DEBUG model decided that index {idx} was not suitable")
        except:
            print(f"DEBUG max retries exceeded for index {idx}")

async def filter_all_questions(paragraphs_processed, judged_worthy_for_questions, LLM, output_dir, take_subset=False,):
    if not take_subset:
        tasks = [
            determine_worthy(
                idx,
                p,
                judged_worthy_for_questions,
                LLM,
                output_dir,
            )
            for idx, p in enumerate(paragraphs_processed)
        ]
    else:
        tasks = [
            determine_worthy(
                idx,
                p,
                judged_worthy_for_questions,
                LLM,
                output_dir,
            )
            for idx, p in enumerate(paragraphs_processed[:13])
        ]
    for future in tqdmasyncio.tqdm.as_completed(tasks):
        await future

def identify_duplicates(tuples: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:

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

    matching_questions = [
        q for q_list in question_dict.values() if len(q_list) == 1 for q in q_list
    ]
    selected_from_duplicates = [
        q_list[0] for q_list in question_dict.values() if len(q_list) > 1
    ]

    return matching_questions + selected_from_duplicates

def fix_text(to_replace_arr, text):
    for startup in to_replace_arr:
        text = text.replace(startup[0], startup[1])
    return text

async def ensure_multiple_answers_are_same(
    info, conv, LLM, assistant_mode
):  # why is this a whole separate function? Once upon a time, LLMs were used in validation here, too. But programmatic validation SEEMS to catch the common problems. This is here so that I can add it back in if I have to.
    """Loop to ensure that the answer is consistent in the conversation and in the tuple."""
    retries = 0
    c = conv
    while retries < 2:  # try twice, since multiturn is an expensive operation
        if call_all_processors(
            c[0], info[0]
        ):  # if programmatic validation passes
            return c

        retries += 1
        if retries >= 2:
            return None
        # If we're here, majority of relevance checks failed
        print("----------------\n\n\n\nRETRYING!!!!\n\n\n\n----------------")
        # Broken info is 1) rare and 2) handled by the retry limit. We don't want to waste compute on regenerating info as they take time.
        retry = await make_multiturn_conversation(info, LLM, assistant_mode)
        if retry is not None:  # Note: retry CANNOT actually be None
            c = retry
        else:
            # If we failed to generate a retry, don't waste compute
            return None

    return None

async def make_multiturn_conversation(info, LLM, assistant_mode):
    conv, conv_output = await multi_turn_conversation(
        info[0],
        info[1],
        info[2],
        info[3],
        LLM,
        assistant_mode=assistant_mode,
    )
    write_output_to_file(conv_output, "./multiturn_conversation_generations", info[4])

    return conv


async def multi_turn_conversation(qatuples, character, scenario, scenario_plan, LLM, assistant_mode=False):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

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
    
    engine_wrapper = LLM['llm']

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
            "scenario": scenario,
            "scenario_plan": scenario_plan,
            "character": character,
            "conv_starter": conv_starter,
            "format_qatuples_qatuples": format_qatuples(qatuples),
        }

        # Load the assistant prompt.
        try:
            cot_prompt, _ = load_external_prompt_and_grammar("multi_turn_conversation_assistant_mode", "dummy_grammar", prompt_content)
        except Exception as e:
            logger.exception(f"An Exception occurred in async 'multi_turn_conversation' function while trying to import its assistant prompt: {e}")

        # Try to load the override LLM dictionary, if it exists. If not, use the function's defaults.
        try: 
            overrides = LLM['override_multi_turn_conversation_assistant_mode_presets']

            # Override the default function presets if it's requested.
            if overrides.get('override_llm_presets') is True:
                logger.info("Overriding default LLM presets for async 'multi_turn_conversation' function.")
                initialized_model = overrides['llm']
                LLM['override_llm_presets'] = True

            # Override the prompt if it's requested.
            if overrides.get('prompt'):
                question_prompt = format_external_text_like_f_string(overrides['prompt'], prompt_content)
                logger.info("Overriding the prompt for async 'multi_turn_conversation' function.")

            # Override the grammar if it's requested.
            if overrides.get('grammar'): 
                questions_grammar = overrides['grammar']
                logger.info("Overriding the grammar for async 'multi_turn_conversation' function.")

        except KeyError:
            logger.info("Overrides for async 'multi_turn_conversation' function not present. Using default presets.")

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
            cot_prompt, _ = load_external_prompt_and_grammar("multi_turn_conversation", "dummy_grammar", prompt_content)
        except Exception as e:
            logger.exception(f"An Exception occurred in async 'multi_turn_conversation' function while trying to import its prompt: {e}")
            
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
                logger.info("Overriding the prompt for async 'multi_turn_conversation' function.")

            # Override the grammar if it's requested.
            if overrides.get('grammar'): 
                questions_grammar = overrides['grammar']
                logger.info("Overriding the grammar for async 'multi_turn_conversation' function.")

        except KeyError:
            logger.info("Overrides for async 'multi_turn_conversation' function not present. Using default presets.")

    # NOTE: Very rarely, the first message of this conv will just be part of the character card, causing the conv to not make much sense. 
    # The cause of this is likely the fact that Elise quotes her character card in her first message.
    # However, referencing the character card in this way also makes characters act as they are described, 
    # which is deemed advantageous enough that I am not changing this for now.
    # I get the sense that LLMs can learn relationships and connections between parts of the prompt, even if they're quite far apart, 
    # if you give them examples like this. It's fascinating to see how each part of the prompt has consequences -- sometimes unintended ones.

    # Note: performance degrades rapidly if you put more than one sentence in a pre-prompt parentheses thing

    if LLM['override_llm_presets']:
        sampling_params = overrides['sampling_params']
    else:
        sampling_params = SamplingParams(
            max_tokens=8000,
            stop=["</s>", "# Input:", "[INST]", "### Instruction", "### Information"],
            temperature=0.5,
            top_k=-1,
            top_p=1,
            min_p=0.6,
        )

    start_time = time.time()
    logger.info(f"Generating 'multi_turn_conversation' completion...")

    completion = await engine_wrapper.submit(cot_prompt, sampling_params)

    end_time = time.time()
    logger.info(f"Done! Completion took {(end_time - start_time) / 60} minutes to generate.")
    logger.info(f"Completion for async 'multi_turn_conversation' function generated. Extracting response pattern...")
    if DEBUG_MODE:
        logger.info(f"\n*** async multi_turn_conversation COMPLETION ***: \n{completion}\n *** async multi_turn_conversation COMPLETION ***\n")

    # Extract plan
    response_pattern = re.compile(
        f"Conversation that answers the provided question \(be sure that you do not change the questions or answers themselves; {charname} will answer the questions, not ask them; the questions and answers provided should be copied word for word, and surrounded by compelling conversation\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )

    generation = response_pattern.search(completion).group(1)
    if DEBUG_MODE:
        logger.info(f"\n*** async multi_turn_conversation GENERATION:***\n\n-------------------\n\n {generation} \n*** async multi_turn_conversation GENERATION: ***\n\n-------------------\n\n")

    # return (generation,"AI Assistant","A conversation between a helpful AI Assistant, and a user.","N/A",qatuples), completion

    return (generation, character, scenario, scenario_plan, qatuples), completion


async def create_info(idx, group, LLM, assistant_mode, multi_turn_convs_info, multi_turn_convs_info_dir, rearrangements_to_take,):
    all_permutations = list(itertools.permutations(group))

    sample_size = min(rearrangements_to_take, len(all_permutations))
    sampled_permutations = random.sample(all_permutations, sample_size)

    group_convs_info = []

    for iter, perm in enumerate(sampled_permutations):
        file_path = os.path.join(multi_turn_convs_info_dir, f"info_{idx}_{iter}.json")

        # Skip if file already exists
        if not os.path.exists(file_path):
            try:
                info = await make_multiturn_conversation_info(perm, LLM, assistant_mode)

                if info is not None:
                    with open(file_path, "w") as file:
                        json.dump(info, file, indent=4)

                group_convs_info.append(info)
            except Exception as e:
                print("ERROR!!!!--!!!!", e)
                traceback.print_exc()
        else:
            print(f"Skipped generating {file_path} as it already exists")

    multi_turn_convs_info.append(group_convs_info)

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

async def create_conversation(
    idx, info, LLM, multi_turn_convs, multi_turn_convs_dir, assistant_mode
):
    file_path = os.path.join(multi_turn_convs_dir, f"conv_{idx}.json")

    # Skip if file already exists
    if not os.path.exists(file_path):
        try:
            conv = await make_multiturn_conversation(
                info, LLM, assistant_mode
            )
            final_conv = await ensure_multiple_answers_are_same(
                info, conv, LLM, assistant_mode
            )

            if final_conv is not None:
                with open(file_path, "w") as file:
                    json.dump(final_conv, file, indent=4)

            multi_turn_convs.append(final_conv)
        except Exception as e:
            print("Had an error, retrying...", e)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            multi_turn_convs.append(data)
        print(f"Skipped generating {file_path} as it already exists")

def convert_directory_to_list(directory_path):
    master_list = []
    simplified_list = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, "r") as file:
                data = json.load(file)
                if isinstance(data, list) and all(
                    isinstance(item, (list, str)) for item in data
                ):
                    master_list.append(data)

                    # Extract and process conversation
                    conversation, primary_char_desc = data[0], data[1]
                    primary_char_name = extract_name(primary_char_desc)
                    dialogues = extract_conversation(
                        conversation
                    )

                    # Convert to simplified format
                    simplified_conversations = []
                    for i, (charname, message) in enumerate(
                        dialogues
                    ):  # Skipping the first message
                        from_person = (
                            "human" if charname == primary_char_name else "gpt"
                        )
                        simplified_conversations.append(
                            {"from": from_person, "value": f"{charname}: {message}"}
                        )

                    if simplified_conversations:  # If there are any conversations
                        simplified_list.append(
                            {"conversations": simplified_conversations}
                        )

    # Write the master list to a new .jsonl file
    with open("master_list.jsonl", "w") as file:
        for item in master_list:
            file.write(json.dumps(item) + "\n")

    # Write the simplified data to a different .jsonl file
    with open("simplified_data.jsonl", "w") as file:
        for item in simplified_list:
            file.write(json.dumps(item) + "\n")

    print(
        "Conversion complete. Master list written to 'master_list.json'. Simplified data written to 'simplified_data.json'."
    )

def convert_directory_and_process_conversations(directory_path):
    master_list = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, "r") as file:
                data = json.load(file)

                if isinstance(data, list) and all(
                    isinstance(item, (list, str)) for item in data
                ):
                    # Extract and process the conversation part
                    conversations = extract_conversation(
                        data[0]
                    )
                    # Convert tuples back to the formatted string as required
                    data[0] = [
                        f"{charname}: {message}" for charname, message in conversations
                    ]
                    master_list.append(data)
                else:
                    print(f"File {filename} is not in the expected format.")

    # Write the master list to a new file
    with open("processed_master_list.json", "w") as file:
        json.dump(master_list, file)

    print(
        "Conversion complete. The processed master list is written to 'processed_master_list.json'."
    )

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
    for i in range(
        2, len(dialogues), 2
    ):  # Answers are at even indices, starting from 2
        if int(i / 2) - 1 >= len(
            qatuples
        ):  # at this point we've reached added stuff that doesn't have a corresponding qatuple
            break
        sequential, comp = has_sequential_chars(
            qatuples[int(i / 2) - 1][1], dialogues[i][1], n
        )
        # print(sequential)
        # print(n)
        if not sequential:
            print(
                f"Answer {int(i/2)}: {dialogues[i][1]} does not match the corresponding answer in qatuples: {qatuples[int(i/2) - 1][1]}, {comp}"
            )
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
    for i in range(
        2, len(dialogues), 2
    ):  # Answers are at even indices, starting from 2
        if int(i / 2) - 1 >= len(
            qatuples
        ):  # at this point we've reached added stuff that doesn't have a corresponding qatuple
            break

        dialogue_answer = dialogues[i][1]
        corresponding_qatuple_answer = qatuples[int(i / 2) - 1][1]
        # Check if the dialogue answer repeats the qatuple answer
        if dialogue_answer.count(corresponding_qatuple_answer) > 1:
            return False
    return True

def check_conversation_length(conv, qatuples):
    """Checks the length of the conversation"""
    # Dialogues with answers should be at even indices that are not 0
    # qatuples are of the format (question, answer,source_text,name_of_text) -- only the first two are used here

    # Get the length of the dialogues
    conv_length = len(conv)

    target_length = len(qatuples) * 2 + 1
    if (
        conv_length < target_length
    ):  # we can have more messages since the AI might add some stuff at the end to wrap up the scene
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
            print(f"Found {string} in the conversation!")
    if matches_found > 2:
        print(
            f"Found {matches_found} matches for strings from the few-shot examples. Validation failed!"
        )
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
        if i // 2 < len(
            qatuples
        ):  # Ensure we only check questions that have corresponding qatuples
            question_from_conv = conv[i][1]
            question_from_tuples = qatuples[i // 2][0]
            # print(question_from_tuples, question_from_conv)
            sequential, _ = has_sequential_chars(
                question_from_tuples, question_from_conv, n
            )
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

    for i in range(2, len(dialogues), 2):  # Answers are at even indices, starting from 2

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
        print("Answers in dialogues do not match corresponding answers in qatuples.")
        return False

    # Check if any dialogue line repeats its corresponding answer
    if not check_for_repeated_dialogue_answers(convs_split, qatuples, 15):
        print("Dialogue line repeats its corresponding answer.")
        return False

    # Check the conversation length
    if not check_conversation_length(convs_split, qatuples):
        print("Conversation is too short! Validation failed!")
        return False

    # Check for text from examples (assuming this is implemented elsewhere)
    if not check_conversation_for_text_from_examples(multiturn_conversation):
        print("Conversation does not contain text from examples. Validation failed!")
        return False

    # Check for unintended repeated quotes
    if not check_for_unintended_repeated_quotes(convs_split, qatuples, 100):
        print("Conversation contains unintended repeated quotes. Validation failed!")
        return False

    # Check each question contains a part of the question from tuples
    result = check_each_question_contains_q_from_tuples(convs_split, qatuples, 15)
    if result is None:
        print(
            "First question does not contain a part of the question from tuples. Validation failed!"
        )
        return None
    elif not result:
        print(
            "Each question does not contain a part of the question from tuples. Validation failed!"
        )
        return False

    # If all checks pass
    return True

def random_name():
    return random.choice(NAMES)

def sanity_check(logic_llm):
    retries = 0
    while retries <= 4:
        decision_prompt = f"""Hi there, """
        # print("DEBUG\n\n" + prompt=decision_prompt)
        completion = llm_call(
            prompt=decision_prompt,
            # max_tokens=100,
            # stop=["</s>", "# Input:", "[INST]","### Instruction"],
            # echo=True,
            # grammar=answer_accurate_grammar,
            temperature=0.2,
        )["choices"][0]["text"]
        # print(completion)

        return

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


class EngineWrapper:
    def __init__(self, model, 
                 api_key=None, 
                 base_url=None, 
                 mode="api", # can be one of api, aphrodite, llama.cpp
                 quantization="gptq", # only needed if using aphrodite mode
                ):
        if mode == "aphrodite":
            engine_args = AsyncEngineArgs(
                model=model,
                quantization=quantization,
                engine_use_ray=False,
                disable_log_requests=True,
                max_model_len=12000,
                dtype="float16"
            )
            self.engine = AsyncAphrodite.from_engine_args(engine_args)
        self.mode = mode
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def submit_completion(self, prompt, sampling_params):  # Submit request and wait for it to stream back fully
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = []
        if "n_predict" not in sampling_params and self.mode == "llamacpp":
            sampling_params["n_predict"] = sampling_params["max_tokens"]

        if DEBUG_MODE:
            logger.info(f"\n\nSETTINGS DUMP\n\nmodel:{self.model}\nprompt:{prompt}\ntemperature:{sampling_params['temperature']}\ntop_p:{sampling_params['top_p']}\nmax_tokens:{sampling_params['max_tokens']}\n")

        if self.mode == "llamacpp":
            return await make_async_api_call(prompt=prompt, sampling_parameters=sampling_params)
        
        if self.mode == "aphrodite":
            aphrodite_sampling_params = SamplingParams(**sampling_params)
            request_id = make_id()
            outputs = []
            # self.engine.add_request(request_id,prompt,sampling_params) #old sync code
            final_output = None
            async for request_output in self.engine.generate(
                prompt, aphrodite_sampling_params, request_id
            ):
                outputs.append(request_output.outputs[0].text)
                final_output = request_output

            # full_output = "".join(outputs)
            return final_output.prompt + final_output.outputs[0].text
        
        if self.mode == "api":
            completion = await self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                stop=sampling_params["stop"],
                max_tokens=sampling_params["max_tokens"],
            )
            completion = completion.choices[0].text
            return prompt + completion
    
    async def submit_chat(self, messages, sampling_params):  # Submit request and wait for it to stream back fully

        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = []
        
        if self.mode == "llamacpp":
            return await make_async_api_call(messages=messages, sampling_parameters=sampling_params)
        elif self.mode == "api":
            messages_cleaned = [
                {
                 "role": message["role"], 
                 "content": message["content"].replace("\\n","\n")
                } for message in messages
            ]
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages_cleaned,
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                stop=sampling_params["stop"],
                max_tokens=sampling_params["max_tokens"],
            )
            completion = completion.choices[0].message.content
            return completion
        else:
            raise Exception("Aphrodite not compatible with chat mode!")











































