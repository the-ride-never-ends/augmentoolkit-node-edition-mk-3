import asyncio
import re
import random
import os
import traceback
import json
import logging
import random
import itertools
import glob
import uuid
import yaml

# Import global variables and functions.
from augmentoolkit import (
    CONCURRENCY_LIMIT,
    COMPLETION_MODE,
    DEBUG_MODE,
    DOUBLE_CHECK_COUNTER,
    PROMPT_DICT,
    PROMPTS,
    DEFAULT_PROMPTS,
    OUTPUT,
    load_external_prompt_and_grammar,
    format_external_text_like_f_string,
    write_output_to_file,
    strip_steps,
    random_name,
    extract_name,
    )


import folder_paths
from custom_nodes.logger import logger

from math import ceil
from typing import Any, List, Tuple, Union

from tqdm import asyncio as tqdmasyncio
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

import matplotlib.pyplot as plt
from collections import Counter


from custom_nodes.augmentoolkit import (
    escape_unescaped_quotes,
    extract_capital_letters,
    extract_name,
    extract_question_answer,
    format_qatuples,
    identify_duplicates,
    process_multiturn_functions,
    run_task_with_limit,
    select_random_capital,
    special_instructions
)

import folder_path

with open('./config.yaml', 'r') as file:
    obj_conf = yaml.safe_load(file)

# Used basically everywhere:
def make_id():
    return str(uuid.uuid4())

augmentoolkit.extract_steps

augmentoolkit.extract_first_words

augmentoolkit.write_output_to_file

# multiturn helpers
# These will probably be used for multiturn rapid-fire answering.

def create_conv_starter(character):
    charname = extract_name.extract_name(character)
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
    return random.choice(conv_starters_filtered)

def create_starting_str(qatuples):
    author_name_letters = extract_capital_letters(qatuples[0][3])
    starting_str = ""
    exclusions = ["X", "Z", "Y", "Q"]
    if author_name_letters:
        starting_str = select_random_capital(exclusions + author_name_letters)
    else:
        starting_str = select_random_capital(exclusions)
    return starting_str

# Idea: use multiple short answers to train the task of answering multiple questions in one response. Like, "Tell me what 2+2 is then tell me who won the battle of Alesia". Two-three short answers per response should be enough.
async def make_multiturn_character(
    qa_tuples, conv_id, assistant_mode=False, character_card_plan_creator=None, character_card_creator=None,completion_mode=None
):
    if (assistant_mode):  # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file
        return "will_be_replaced", "will_be_replaced"
    
    instructions = special_instructions(n=1).strip()
    if not completion_mode:
        instructions = escape_unescaped_quotes(instructions).replace("\n","\\n")
    if completion_mode:
        (
            plan,
            card_plan_output,
        ) = await character_card_plan_creator.generate(
            arguments={
                "textname": qa_tuples[0][3],
                "text": qa_tuples[0][2],
                "question_answer_list": format_qatuples(qa_tuples),
                "special_instructions": instructions
            }
        )  # I will reuse the many tuples function for short question-answers, there's a lot of prompting in here already
    else:
        (
            plan,
            card_plan_output,
        ) = await character_card_plan_creator.generate(
            arguments={
                "textname": qa_tuples[0][3],
                "text": qa_tuples[0][2],
                "question_answer_list": escape_unescaped_quotes(format_qatuples(qa_tuples)).replace("\n","\\n"),
                "special_instructions": instructions
            }
        )
    write_output_to_file(card_plan_output, OUTPUT + "/multiturn_card_plan_generations", conv_id)
    
    starting_str = create_starting_str(qa_tuples)
    (
        char,
        char_output,
    ) = await character_card_creator.generate(
        arguments={
            "text": qa_tuples[0][2],
            "textname": qa_tuples[0][3],
            "special_instructions": instructions,
            "plan": plan,
            "starting_str": starting_str
            
        }
    )  # creates a character card
    write_output_to_file(char_output, OUTPUT + "/multiturn_card_generations", conv_id)
    return char, instructions


async def make_multiturn_scenario(
    qa_tuples, character, conv_id, assistant_mode=False, scenario_plan_creator=None, scenario_creator=None, completion_mode=None
):
    if (
        assistant_mode
    ):  # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file
        return "will_be_replaced", "will_be_replaced"
    if completion_mode:
        (
            plan,
            scenario_plan_output,
        ) = await scenario_plan_creator.generate(
            arguments={
                "question_answer_list": format_qatuples(qa_tuples),
                "character": character,
            }
        )
    else:
        (
            plan,
            scenario_plan_output,
        ) = await scenario_plan_creator.generate(
            arguments={
                "question_answer_list": escape_unescaped_quotes(format_qatuples(qa_tuples)).replace("\n","\\n"),
                "character": character,
            }
        )
    
    plan = fix_scenario_plan(plan, character)
    write_output_to_file(
        scenario_plan_output, OUTPUT + "/multiturn_scenario_plan_generations", conv_id
    )
    
    variation = select_variation(character)
    if completion_mode:
        (
            scenario,
            scenario_output,
        ) = await scenario_creator.generate(
            arguments={
                "question_answer_list": format_qatuples(qa_tuples),
                "character": character,
                "plan": plan,
                "selected_variation": variation
            }
        )  # creates a scenario based on a character card and question/answer tuple
    else:
        (
            scenario,
            scenario_output,
        ) = await scenario_creator.generate(
            arguments={
                "question_answer_list": escape_unescaped_quotes(format_qatuples(qa_tuples)).replace("\n","\\n"),
                "character": character,
                "plan": plan,
                "selected_variation": variation
            }
        )
    write_output_to_file(scenario_output, OUTPUT + "/multiturn_scenario_generations", conv_id)
    return scenario, plan


async def make_multiturn_conversation_info(
    qa_tuples, 
    assistant_mode=False, 
    character_card_plan_creator=None, 
    character_card_creator=None, 
    scenario_plan_creator=None, 
    scenario_creator=None, 
    completion_mode=None
):
    conv_id = make_id()
    if (
        assistant_mode
    ):  # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file
        return (qa_tuples, "will", "be", "replaced", conv_id)
    # thought_plan = create_thought_plan_many_tuples(qa_tuples,character,scenario,logic_llm) # There IS a way to make multiturn chain of thought answering work: generate each pair of messages using a separate prompt or a separate function, each of which has only the thought plan for that question/answer pair. But simply cramming in all the step-by-step things will confuse the hell out of the poor model. So for the first release version we're skipping it and just giving the response, with no reasoning, in the multiturn convs.
    retries = 0
    done = False
    while not done and retries < 3:
        retries = retries + 1
        character, instructions = await make_multiturn_character(
            qa_tuples, 
            conv_id, 
            assistant_mode=assistant_mode, 
            character_card_plan_creator=character_card_plan_creator, 
            character_card_creator=character_card_creator, 
            completion_mode=completion_mode
        )
        if "What's your backstory?" not in character:
            logger.warning("Failed to properly generate card, retrying")
            continue
        done = True
    scenario, scenario_plan = await make_multiturn_scenario(
        qa_tuples, 
        character, 
        conv_id, 
        assistant_mode=assistant_mode, 
        scenario_plan_creator=scenario_plan_creator, 
        scenario_creator=scenario_creator, 
        completion_mode=completion_mode
    )

    return (qa_tuples, character, scenario, scenario_plan, conv_id)


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
        identify_duplicates(group)
        for group in list(groups.values())
    ]


def extract_reasoning_from_context_check(response):
    decision_pattern = re.compile(r"Final judgment:(.+)", re.IGNORECASE)
    determination = decision_pattern.search(response).group(1).strip()
    if "pass" in determination.lower():
        print("Leaving be...")
        return (True, response)#, completion
    elif "reword" in determination.lower():
        print("Rewording...")
        q, a = extract_question_answer.extract_question_answer(response)
        print((q, a))
        return (q,a)#(q, a, qatuple[2], qatuple[3]), completion
    elif "fail" in determination.lower():
        print("Setting to None...")
        return (False, response)#, completion
    else:
        print("Did not contain relevant or irrelevant! Retrying")

# Postprocessing function for question/answer validation
async def repair_qatuple_context(
    idx, tup, engine_wrapper, writepath, vetted_qa_tuples, use_filenames=False, completion_mode=None, logging_level=logging.INFO
):
    # NOTE set up the generation step
    context_repairer_path = "check_qatuple_context_no_filenames"
    if use_filenames:
        context_repairer_path = "check_qatuple_context_filenames"

    if completion_mode:
        context_repairer_path = context_repairer_path + ".txt"
    else:
        context_repairer_path = context_repairer_path + ".json"
        
    repair_context_regex = re.compile(
                r"Reasoning and thought process \(be thorough\):(.+)",
                re.DOTALL | re.IGNORECASE,
            )
    context_repairer = GenerationStep(
        prompt_path=context_repairer_path,
        regex=repair_context_regex,
        sampling_params={
            "max_tokens": 2000,
            "stop": [
                "### Response",
                "\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
            ],
            "temperature": 0.2,
        },
        completion_mode=completion_mode,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level,
        output_processor=extract_reasoning_from_context_check,
        prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
    )

    # Resume normal control flow
    file_path = os.path.join(writepath, f"revised_{idx}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()  # Read the file once and store its content
            print(file_path)
            if content == "failed":
                print("Loaded failed file")
                vetted_qa_tuples[idx] = None
                return None
            print("Loaded file:")
            print(content)
            try:
                data = json.loads(content)  # Convert the string back to JSON
                vetted_qa_tuples[idx] = (data[0], data[1], data[2], data[3])
                return None
            except json.JSONDecodeError:
                print("JSON decode error with the contents:", content)

    try:
        revision_id = make_id()
        revision, revision_output = await context_repairer.generate(
            arguments={
                "textname": tup[3],
                "question": tup[0],
                "answer": tup[1],
            }
        )
        write_output_to_file(
            revision_output, OUTPUT + "/question_context_revision_generations", revision_id
        )  # incidentally, identifying the problem and fixing it in the same step (without another planning step) works a lot better than identifying it and then trying to fix it in the next step.
        if isinstance(revision[0], str):  # if the thing was reworded
            vetted_qa_tuples[idx] = (revision[0], revision[1], tup[2], tup[3]) # replace the old tuple with the new one, revision doesn't have text name so we keep the old one
        elif not revision[0]:
            vetted_qa_tuples[
                idx
            ] = None  # prepare item for deletion later; right now we just store it as None because indexes
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
        print("!!! ERROR!", e)
        traceback.print_exc()

def parse_answer_accuracy_validation(response):
    determination_pattern = re.compile(
        r"Overall Accuracy Determination:(.+)", re.DOTALL
    )
    determination = determination_pattern.search(response).group(1).strip()
    if (
                "inaccurate" in determination.lower()
                or "Inaccurate" in determination.lower()
                or "mostly" in determination.lower()
                or "partial" in determination.lower() or "irrelevant" in determination.lower()
            ):  # The "mostly" is there to catch "mostly accurate" which the model says occasionally, and which actually means inaccurate.
                return (False, response)
    elif (
        "accurate" in determination.lower()
    ):
        return (True, response)
    else:
        logging.ERROR("Answer accuracy validation made a mistake")
        raise Exception("answer accuracy validation did not include a judgement")

# Control flow helpers -- Question/Answer Validation
async def vet_answer_accuracy_loop(
    qa_tuple,
    total_retries,
    run_id,
    engine_wrapper=None,
    double_check_counter=3,
    use_filenames=False,
    completion_mode=None,
    logging_level=None,
    new_q_generator=None
):
    
    # NOTE Set up answer check generation step
    prompt_path_ans_accuracy_check = "check_answer"
    if completion_mode:
        prompt_path_ans_accuracy_check = prompt_path_ans_accuracy_check + ".txt"
    else:
        prompt_path_ans_accuracy_check = prompt_path_ans_accuracy_check + ".json"
    check_ans_accuracy_regex = re.compile(
                r"Reasoning and thought process \(the text is your single source of truth\):\n(.+)",
                re.DOTALL,
            )

    answer_accuracy_checker = GenerationStep(
        prompt_path=prompt_path_ans_accuracy_check,
        regex=check_ans_accuracy_regex,
        sampling_params={
                "max_tokens": 6000,
                "stop": [
                    "### Response",
                    "\n\n\n\n\n",
                    "</s>",
                    "# Input:",
                    "[INST]",
                    "### Instruction",
                    "[INST",
                ],
                "temperature": 0.2,
            },
        completion_mode=completion_mode,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level,
        output_processor=parse_answer_accuracy_validation,
        prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
    )
    
    # Resume normal control flow code
    
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

            judgement, answer_accuracy_output = await answer_accuracy_checker.generate(
                arguments={
                    "text": qtuple[2],
                    "question": qtuple[0],
                    "answer": qtuple[1]
                }
            )
            write_output_to_file(
                answer_accuracy_output, OUTPUT + "/check_answer_accuracy_generations", run_id
            )
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
            para = qtuple[2]
            para_name = qtuple[3]
            (
                qtuple_partial,
                generate_new_q_output,
            ) = await new_q_generator.generate(
                arguments={
                    "textname": qtuple[3],
                    "text": qtuple[2]
                }
            )
            qtuple = (qtuple_partial[0],qtuple_partial[1],para,para_name)
            write_output_to_file(
                generate_new_q_output, OUTPUT + "/regenerate_question_generations", run_id
            )
            return await vet_question_loop(
                qtuple,
                total_retries,
                question_group_id=run_id.split("--subquestion--")[0],
                engine_wrapper=engine_wrapper,
                double_check_counter=double_check_counter,
                use_filenames=use_filenames,
                completion_mode=completion_mode,
                logging_level=logging_level
            )  # going to get one hell of a call stack by the end of this, but it should be fine
    except Exception as e:
        print("!!ERROR!!")
        print(e)
        traceback.print_exc()

    return (None, None, None, qtuple[3])

def parse_answer_relevancy_validation_step(thought_process):
    judgement_pattern = re.compile(
        r"Explanation of Judgment:(.+)", re.DOTALL | re.IGNORECASE
    )
    determination = judgement_pattern.search(thought_process).group(1).strip()
    if (
        "irrelevant" in determination.lower()
        or "mostly" in determination.lower()
        or "partial" in determination.lower()
        or "introduces information not present in the text"
        in determination.lower()
    ):  # Hack to get around faulty outputs
        return (False, thought_process)#, completion
    elif "relevant" in determination or "Relevant" in determination:
        return (True, thought_process)#, completion
    else:
        logging.ERROR(f"Answer relevancy parsing failed! Retrying! {judgement_pattern}")
        raise Exception("error in judgement extranction (ans relevancy)")

async def vet_answer_relevance_loop(
    qa_tuple,
    total_retries,
    run_id,
    engine_wrapper=None,
    double_check_counter=3,
    use_filenames=False,
    completion_mode=None,
    logging_level=None,
    new_q_generator=None, # we pass the new q generator around so the code is less cluttered
):
    
    # NOTE Set up answer check generation step
    prompt_path_ans_relevancy_check = "check_answer_relevancy_with_text"
    check_ans_relevancy_regex = re.compile(
                r"Reasoning and thought process \(be careful about extra details, even vague ones\):\n(.+)",
                re.DOTALL | re.IGNORECASE,
            )
    
    if completion_mode:
        prompt_path_ans_relevancy_check = prompt_path_ans_relevancy_check + ".txt"
    else:
        prompt_path_ans_relevancy_check = prompt_path_ans_relevancy_check + ".json"

    answer_relevancy_checker = GenerationStep(
        prompt_path=prompt_path_ans_relevancy_check,
        regex=check_ans_relevancy_regex,
        sampling_params={
                "max_tokens": 5500,
                "stop": [
                    "### Response",
                    "\n\n\n\n\n",
                    "</s>",
                    "# Input:",
                    "[INST]",
                    "### Instruction",
                    "[INST",
                ],
                "temperature": 0.2,
            },
        completion_mode=completion_mode,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level,
        output_processor=parse_answer_relevancy_validation_step,
        prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
    )
    
    # Resume normal control flow code
    try:
        qtuple = qa_tuple
        # print(
        # f"\n\nStarting RELEVANCE loop for question: {qtuple[0]}, context: {qtuple[2]}"
        # )
        passed_checks = 0
        times_checked = 0
        dissenting_reasoning = ""
        while times_checked < double_check_counter:
            # print(
            # f"\n\nRELEVANCE CALL CHECK ANSWER: {qtuple[0]}, context: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}"
            # )
            (
                judgement,
                answer_relevancy_output,
            ) = await answer_relevancy_checker.generate(
                arguments={
                    "text": qtuple[2],
                    "question": qtuple[0],
                    "answer": qtuple[1]
                }
            )
            write_output_to_file(
                answer_relevancy_output, OUTPUT + "/check_answer_relevancy_generations", run_id
            )
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
            # print(f"\n\nRELEVANCE CHECKS PASSED")
            return await vet_answer_accuracy_loop(
                qtuple,
                total_retries,
                run_id,
                engine_wrapper=engine_wrapper,
                double_check_counter=double_check_counter,
                use_filenames=use_filenames,
                completion_mode=completion_mode,
                logging_level=logging_level,
                new_q_generator=new_q_generator
            )
        else:
            # print(f"\n\nRELEVANCE CHECKS FAILED - SENDING BACK TO QUESTION LOOP")
            total_retries += 1
            para = qtuple[2]
            para_name = qtuple[3]
            (
                qtuple_partial,
                generate_new_q_output,
            ) = await new_q_generator.generate(
                arguments={
                    "textname": qtuple[3],
                    "text": qtuple[2]
                }
            )
            print(qtuple_partial)
            qtuple = (qtuple_partial[0],qtuple_partial[1],para,para_name)
            write_output_to_file(
                generate_new_q_output, OUTPUT + "/regenerate_question_generations", run_id
            )
            return await vet_question_loop(
                qtuple,
                total_retries,
                question_group_id=run_id.split("--subquestion--")[0],
                engine_wrapper=engine_wrapper,
                double_check_counter=double_check_counter,
                use_filenames=use_filenames,
                completion_mode=completion_mode,
                logging_level=logging_level
            )
    except Exception as e:
        print("!!ERROR!!")
        print(e)
        traceback.print_exc()

    return (None, None, None, qtuple[3])

def parse_validation_step(response):
    decision_pattern = re.compile(
                r"Final Judgment:(.+)", re.DOTALL | re.IGNORECASE
            )
    determination = decision_pattern.search(response).group(1).strip()
    if (
        "irrelevant" in determination
        or "Irrelevant" in determination.lower()
        or "mostly" in determination.lower()
        or "partial" in determination.lower()
        or "introduces information not present in the text"
        in determination.lower()
        ):
        return (False, response) # TODO ensure that in the control flow code it passes on (False, response), completion
    elif "relevant" in determination or "Relevant" in determination:
        return (True, response) # TODO same as above(True, response), completion
    else:
        logging.ERROR("Did not contain relevant or irrelevant! Retrying")
        raise Exception("Validation step screwed up and did not reach a conclusion! Retrying!")

async def vet_question_loop(
    qa_tuple,
    total_retries,
    question_group_id=None,
    engine_wrapper=None,
    double_check_counter=3,
    use_filenames=False,
    completion_mode=None,
    logging_level=None,
    
):
    # NOTE Set up question check generation step
    prompt_path_q_check = "check_question"
    check_q_regex = re.compile(
                r"Reasoning and thought process \(be careful around \"how\" and \"why\" questions\):(.+)",
                re.DOTALL | re.IGNORECASE,
            )

    if completion_mode:
        prompt_path_q_check = prompt_path_q_check + ".txt"
    else:
        prompt_path_q_check = prompt_path_q_check + ".json"

    question_checker = GenerationStep(
        prompt_path=prompt_path_q_check,
        regex=check_q_regex,
        sampling_params={
                "max_tokens": 4000,
                "stop": [
                    "### Response",
                    "\n\n\n\n\n",
                    "</s>",
                    "# Input:",
                    "[INST]",
                    "### Instruction",
                    "[INST",
                ],
                "temperature": 0.2,
            },
        completion_mode=completion_mode,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level,
        output_processor=parse_validation_step,
        prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
    )
    
    # NOTE Set up generate new question step
    prompt_path_new_q_gen = "new_q_gen_no_filenames"
    if use_filenames:
        prompt_path_new_q_gen = "new_q_gen_filenames"
    
    new_q_gen_regex = re.compile(
        r"Question \(based on text\):\n(.+)", re.IGNORECASE | re.DOTALL
    )
    
    if completion_mode:
        prompt_path_new_q_gen = prompt_path_new_q_gen + ".txt"
    else:
        prompt_path_new_q_gen = prompt_path_new_q_gen + ".json"
    
    if completion_mode:
        new_q_generator = GenerationStep(
            prompt_path=prompt_path_new_q_gen,
            regex=new_q_gen_regex,
            sampling_params={
                    "max_tokens": 3000,
                    "stop": [
                        "### Response",
                        "\n\n\n\n\n",
                        "</s>",
                        "# Input:",
                        "[INST]",
                        "### Instruction",
                        "[INST",
                    ],
                    "temperature": 0.2,
                },
            completion_mode=completion_mode,
            retries=3,
            engine_wrapper=engine_wrapper,
            logging_level=logging_level,
            output_processor=extract_question_from_response_completionmode,
            prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
        )
    else:
        new_q_generator = GenerationStep(
            prompt_path=prompt_path_new_q_gen,
            regex=new_q_gen_regex,
            sampling_params={
                    "max_tokens": 3000,
                    "stop": [
                        "### Response",
                        "\n\n\n\n\n",
                        "</s>",
                        "# Input:",
                        "[INST]",
                        "### Instruction",
                        "[INST",
                    ],
                    "temperature": 0.2,
                },
            completion_mode=completion_mode,
            retries=3,
            engine_wrapper=engine_wrapper,
            logging_level=logging_level,
            output_processor=extract_question_from_response_chatmode,
            prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
        )
    
    # Resume normal control flow code
    try:
        qtuple = qa_tuple
        # print(
        #     f"\n\nStarting QUESTION loop for question: {qtuple[0]}, context: {qtuple[2]}"
        # )
        while total_retries <= 4:
            run_id = question_group_id + "--subquestion--" + make_id()
            passed_checks = 0
            times_checked = 0
            dissenting_reasoning = ""
            while times_checked < double_check_counter:
                # print(
                #     f"\n\nQUESTION CALL CHECK ANSWER: {qtuple[0]}, context: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}"
                # )
                judgement, check_q_output = await question_checker.generate(
                    arguments={
                        "text": qtuple[2],
                        "question": qtuple[0]
                    }
                )
                
                # Now we need to put the judgement together into the format it expects it to be in
                
                write_output_to_file(
                    check_q_output, OUTPUT + "/check_question_generations", run_id
                )
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

            if passed_checks >= ceil(
                double_check_counter / 2
            ):  # if all question checks passed
                # print(f"\n\nQUESTION CHECKS PASSED retries: {total_retries}")
                return await vet_answer_relevance_loop(
                    qtuple,
                    total_retries,
                    run_id,
                    engine_wrapper=engine_wrapper,
                    double_check_counter=double_check_counter,
                    use_filenames=use_filenames,
                    new_q_generator=new_q_generator,
                    completion_mode=completion_mode,
                    logging_level=logging_level
                )
            else:
                # Generate new question and restart the loop
                # print(
                #     f"\n\nQUESTION CHECKS FAILED - GENERATING NEW QUESTION retries: {total_retries}"
                # )
                total_retries += 1
                if (
                    total_retries <= 4
                ):  # only regen question if we're not already at max regens
                    para = qtuple[2]
                    para_name = qtuple[3]
                    (
                        qtuple_partial,
                        generate_new_q_output,
                    ) = await new_q_generator.generate(
                        arguments={
                            "textname": qtuple[3],
                            "text": qtuple[2]
                        }
                    )
                    qtuple = (qtuple_partial[0],qtuple_partial[1],para,para_name)
                    write_output_to_file(
                        generate_new_q_output,
                        OUTPUT + "/regenerate_question_generations",
                        run_id,
                    )
                    print("New question: ", qtuple)
                # no calling of vet_question_loop, since we're already in a while loop
    except Exception as e:
        print("!!ERROR!!")
        print(e)
        traceback.print_exc()

    return (None, None, None, qtuple[3])


def extract_questions_from_response_completionmode(generation):
    questions = []
    if DEBUG_MODE:
        logger.info(f"!! Model Output: !!\n{generation}")

    pattern = re.compile(
                r"(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)",
                re.DOTALL | re.MULTILINE | re.IGNORECASE,
            )
    matches = pattern.findall(generation)

    if len(matches) == 0: # Because of how the generate step class is structured, this raise will cause a retry, as the original did. No it's not using an exception for normal control flow, if the llm screwed up that's an error.
        raise Exception("Failed to generate questions!") 

    for match in matches:
        questions.append(
            (
                match[0].replace(") ", "", 1).strip(),
                match[1].replace(") ", "", 1).strip(),
                # para_tuple[0].replace(") ", "", 1), # These have to get added in the control flow, minus the .replace() that's actually wrong
                # para_tuple[1].replace(") ", "", 1),
            )
        )
    if DEBUG_MODE:
        logger.info(f"\n\n\nExtract questions from response DEBUG!!!\n{questions}")

    return questions

def extract_questions_from_response_chatmode(generation): # TODO extract to non-controlflow file
    print(generation)
    questions = []
    if DEBUG_MODE:
        logger.info(f"!! Model Output: !!\n{generation}")

    pattern = re.compile(
                r"\d+\.\) (.*?)\\nAnswer: (.*?)(?=\\n\\n|\Z)",
                re.DOTALL | re.MULTILINE | re.IGNORECASE,
            )
    matches = pattern.findall(generation+"\\n\\n")

    if len(matches) == 0:
        raise Exception("Failed to generate questions!") # Because of how the generate step class is structured, this raise will cause a retry, as the original did. No it's not using an exception for normal control flow, if the llm screwed up that's an error.

    for match in matches:
        questions.append(
            (
                match[0].replace(") ", "", 1).strip(),
                match[1].replace(") ", "", 1).strip(),
                # para_tuple[0].replace(") ", "", 1), # These have to get added in the control flow, minus the .replace() that's actually wrong
                # para_tuple[1].replace(") ", "", 1),
            )
        )
    if DEBUG_MODE:
        logger.info(f"\n\n\nExtract questions from response DEBUG!!!\n{questions}")

    return questions

def extract_question_from_response_completionmode(generation): # TODO extract to non-controlflow file
    questions = []

    pattern = re.compile(
                r"(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)",
                re.DOTALL | re.MULTILINE | re.IGNORECASE,
            )
    matches = pattern.findall(generation)

    if len(matches) == 0:
        raise Exception("Failed to generate questions!") # Because of how the generate step class is structured, this raise will cause a retry, as the original did. No it's not using an exception for normal control flow, if the llm screwed up that's an error.

    for match in matches:
        if DEBUG_MODE:
            logger.info(f"\n\n\nExtract questions from response DEBUG!!!\n{questions}")

        return (
                match[0].replace(") ", "", 1).strip(),
                match[1].replace(") ", "", 1).strip(),
                # para_tuple[0].replace(") ", "", 1), # These have to get added in the control flow, minus the .replace() that's actually wrong
                # para_tuple[1].replace(") ", "", 1),
            )
        
def extract_question_from_response_chatmode(generation): # TODO extract to non-controlflow file
    pattern = re.compile(
                r"\d+\.?\)?:? (.*?)\\nAnswer: (.*?)(?=\\n\\n|\Z)",
                re.DOTALL | re.MULTILINE | re.IGNORECASE,
            )
    matches = pattern.findall(generation+"\\n\\n")

    if len(matches) == 0:
        raise Exception("Failed to generate questions!") # Because of how the generate step class is structured, this raise will cause a retry, as the original did. No it's not using an exception for normal control flow, if the llm screwed up that's an error.

    for match in matches:
        if DEBUG_MODE:
            logger.info(f"\n\n\nExtract questions from response DEBUG!!!\n{match}") # Maybe this is right???

        return (
                match[0].replace(") ", "", 1).strip(),
                match[1].replace(") ", "", 1).strip(),
                # para_tuple[0].replace(") ", "", 1), # These have to get added in the control flow, minus the .replace() that's actually wrong
                # para_tuple[1].replace(") ", "", 1),
            )

# Question generation ASDF
async def generate_qatuples_from_para(
    idx,
    para,
    engine_wrapper=None,
    vetted_qa_tuples=None,
    qa_tuples_dir=None,
    double_check_counter=3,
    use_filenames=False,
    completion_mode=None,
    logging_level=None
):

    # NOTE Set up qatuple plan generation step #
    prompt_path_qatuples_plan = "qatuples_plan_no_filenames"
    if use_filenames:
        prompt_path_qatuples_plan = "qatuples_plan_filenames"
    
    qatuples_plan_regex = re.compile(
        r"Reasoning and thought process \(being careful to only plan questions that are entirely based on the text provided\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    
    if completion_mode:
        prompt_path_qatuples_plan = prompt_path_qatuples_plan + ".txt"
    else:
        prompt_path_qatuples_plan = prompt_path_qatuples_plan + ".json"

    qatuples_planner = GenerationStep(
        prompt_path=prompt_path_qatuples_plan,
        regex=qatuples_plan_regex,
        sampling_params={
        "max_tokens": 3000,
        "stop": [
            "### Response",
            "\n\n\n\n\n",
            "</s>",
            "# Input:",
            "[INST]",
            "### Instruction",
            "[INST",
            "Text to plan questions from"
        ],
        "temperature": 0.8,
        # top_k=-1,
        "top_p": 1,
        # min_p=0.5,
        },
        completion_mode=completion_mode,
        retries=0,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level,
        prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
    )
    
    # NOTE Set up qatuple generation step #
    
    prompt_path_qatuples_gen = "qatuples_gen_no_filenames"
    if use_filenames:
        prompt_path_qatuples_gen = "qatuples_gen_filenames"
    
    if completion_mode:
        prompt_path_qatuples_gen = prompt_path_qatuples_gen + ".txt"
    else:
        prompt_path_qatuples_gen = prompt_path_qatuples_gen + ".json"
    
    qatuples_gen_regex = re.compile(
        r"Questions \(make 4\):\n(.+)", re.IGNORECASE | re.DOTALL
    )
    if completion_mode:
        qatuples_generator = GenerationStep(
            prompt_path=prompt_path_qatuples_gen,
            regex=qatuples_gen_regex,
            sampling_params={
                    "max_tokens": 2000,
                    "stop": [
                        "### Response",
                        "\n\n\n\n\n",
                        "</s>",
                        "# Input:",
                        "[INST]",
                        "### Instruction",
                        "[INST",
                    ],
                    "temperature": 0.8,
                    # top_k=-1,
                    "top_p": 1,
                    # min_p=0.5,
                },
            completion_mode=completion_mode,
            retries=3,
            engine_wrapper=engine_wrapper,
            logging_level=logging_level,
            output_processor=extract_questions_from_response_completionmode,
            prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
        )
    else:
        qatuples_generator = GenerationStep(
            prompt_path=prompt_path_qatuples_gen,
            regex=qatuples_gen_regex,
            sampling_params={
                    "max_tokens": 2000,
                    "stop": [
                        "### Response",
                        "\n\n\n\n\n",
                        "</s>",
                        "# Input:",
                        "[INST]",
                        "### Instruction",
                        "[INST",
                    ],
                    "temperature": 0.8,
                    # top_k=-1,
                    "top_p": 1,
                    # min_p=0.5,
                },
            completion_mode=completion_mode,
            retries=3,
            engine_wrapper=engine_wrapper,
            logging_level=logging_level,
            output_processor=extract_questions_from_response_chatmode,
            prompt_folder=PROMPTS,
            default_prompt_folder=DEFAULT_PROMPTS
        )
    # Resume normal control flow code
    try:
        existing_files = glob.glob(
            os.path.join(qa_tuples_dir, f"para_{idx}_*.json")
        )  # check if qs already exist

        if len(existing_files) > 0:  # If files exist, skip this paragraph entirely
            print(f"Skipping para_{idx} as files already exist; loading said files")
            for file_path in existing_files:
                with open(file_path, "r") as file:
                    qa_tuple = tuple(json.load(file))
                vetted_qa_tuples.append(qa_tuple)
            return
        question_group_id = make_id()
        if DEBUG_MODE:
            logger.info(f"\n\n\nOUTER LOOP CALL GENERATE QPLAN para: {para}, \n\n idx: {idx}")
        (
            plan,
            questions_plan_output,
        ) = await qatuples_planner.generate(
            arguments={
                "textdetails": para[1],
                "text": para[0]
            }
        )
        write_output_to_file(
            questions_plan_output, OUTPUT + "/question_plan_generations", question_group_id
        )
        if DEBUG_MODE:
            logger.inf(f"\n\n\nOUTER LOOP CALL GENERATE Q: {para}, \n\n idx: {idx} \n\n plan: {plan}")

        (
            question_answer_tuples,
            question_generation_output,
        ) = await qatuples_generator.generate(
            arguments={
                "text": para[0],
                "textdetails": para[1],
                "plan": strip_steps.strip_steps(plan)
            }
        )
        
        question_answer_tuples_more_info = [(qatup[0],qatup[1],para[0],para[1]) for qatup in question_answer_tuples]
        write_output_to_file(
            question_generation_output,
            OUTPUT + "/question_generation_generations",
            question_group_id,
        )
        for qnum, question_answer_tuple in enumerate(question_answer_tuples_more_info):
            if DEBUG_MODE:
                logger.info(f"\n\n=======!!=BEGIN VETTING QA TUPLE {idx}_{qnum}=!!=======\n\n")
            good_qa_tuple = await vet_question_loop(
                question_answer_tuple,
                0,
                question_group_id=question_group_id,
                engine_wrapper=engine_wrapper,
                double_check_counter=double_check_counter,
                use_filenames=use_filenames,
                completion_mode=completion_mode,
                logging_level=logging_level
            )

            # Write resulting question file if the tuple is not None
            if good_qa_tuple[0] is not None:
                file_path = os.path.join(qa_tuples_dir, f"para_{idx}_q_{qnum}.json")
                with open(file_path, "w") as file:
                    json.dump(good_qa_tuple, file, indent=4)

            vetted_qa_tuples.append(
                good_qa_tuple
            )  # We must filter out all None values at the end; but appending Nones lets us know where things went wrong, and how often.
    except Exception as e:
        print(f"Q ERROR: {e}")
        traceback.print_exc()


# Graphing code generated by GPT-4. May be suboptimal/ugly.
def filter_and_graph(tuples,graph):
    # Count the occurrences of None and non-None for each source text
    source_counts = Counter()
    for paragraph, source in tuples:
        if paragraph is None:
            source_counts[source] = source_counts.get(source, [0, 0])
            source_counts[source][0] += 1
        else:
            source_counts[source] = source_counts.get(source, [0, 0])
            source_counts[source][1] += 1
    if graph:
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
async def determine_worthy(
    idx,
    p,
    judged_worthy_for_questions,
    output_dir,
    judge: GenerationStep,
):
    # for idx, p in tqdm(enumerate(paragraphs_processed[:10])):
    file_name = f"{idx}.json"
    file_path = os.path.join(output_dir, file_name)
    # Check if the judgement for this paragraph already exists
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
            print("LOADING: ", data)
        if isinstance(data, str):
            judged_worthy_for_questions.append((None, data[7:])) # hacky way of appending only the text name. See the file output of a failed judgement for details (Takes after "failed|")
        else:
            judged_worthy_for_questions.append((data["paragraph"], data["metadata"]))
    else:
        judgement = await judge.generate(
            arguments={
                "text": p[0],
                "textname": p[1]
            }
        )
        to_append = (None,p[1])
        if judgement:
            to_append = (p[0],p[1])
            
        judged_worthy_for_questions.append(to_append)

        # Prepare the data to be written to the file
        if judgement:
            # The paragraph passed the judgement
            data_to_write = {"paragraph": to_append[0], "metadata": to_append[1]}
        else:
            # The paragraph did not pass the judgement
            data_to_write = f"failed|{to_append[1]}"

        # Write the judgement to a unique file as JSON
        with open(file_path, "w") as file:
            json.dump(data_to_write, file)

        # Debug messages
        try:
            if judgement:
                print(f"DEBUG model decided that index {idx} was suitable")
            else:
                print(f"DEBUG model decided that index {idx} was not suitable")
        except:
            print(f"DEBUG max retries exceeded for index {idx}")

def judge_paragraph_processor(determination): # TODO extract to separate file to avoid muddying the control flow code
    if "unsuitable" in determination.lower():
        return False # control flow has been modified to use the information it has, based on the determination of the output processors
    elif "suitable" in determination.lower():
        return True

# EXEMPLAR
async def filter_all_questions(
    paragraphs_processed,
    judged_worthy_for_questions,
    engine_wrapper,
    output_dir,
    take_subset=False,
    use_filenames=False,
    rtwl=None,
    completion_mode=None,
    logging_level=None
):
    
    if use_filenames:
        prompt_path = "judge_paragraph_filenames"
    else:
        prompt_path = "judge_paragraph_no_filenames"

    judgement_regex = re.compile(
            r"Reasoning and thought process \(reason intelligently\):(.+)", re.DOTALL | re.IGNORECASE,)
    
    if completion_mode:
        prompt_path = prompt_path + ".txt"
    else:
        prompt_path = prompt_path + ".json"
    
    judge = GenerationStep(
        prompt_path=prompt_path,
        regex=judgement_regex,
        sampling_params={
            "max_tokens": 2000,
            # "min_p": 0.4,
            "stop": [
                "### Response",
                "\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
            ],
            "temperature": 0.2,
        },
        completion_mode=completion_mode,
        retries=2,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level, # TODO change to warning
        output_processor=judge_paragraph_processor,
        return_input_too=False,
        prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
    )
    if not take_subset:
        tasks = [
            determine_worthy(
                idx,
                p,
                judged_worthy_for_questions,
                output_dir,
                judge
            )
            for idx, p in enumerate(paragraphs_processed)
        ]
    else:
        tasks = [
            determine_worthy(
                idx,
                p,
                judged_worthy_for_questions,
                output_dir,
                judge
            )
            for idx, p in enumerate(paragraphs_processed[:13])
        ]
    limited_tasks = [rtwl(task) for task in tasks]
    for future in tqdmasyncio.tqdm.as_completed(limited_tasks):
        await future


augmentoolkit.ChunkSentence.sentence_chunking_algorithm

augmentoolkit.fix_text


async def ensure_multiple_answers_are_same(
    info, conv, multi_turn_conv_generator,completion_mode=None
):  # why is this a whole separate function? Once upon a time, LLMs were used in validation here, too. But programmatic validation SEEMS to catch the common problems. This is here so that I can add it back in if I have to.
    """Loop to ensure that the answer is consistent in the conversation and in the tuple."""
    retries = 0
    c = conv
    while retries < 2:  # try twice, since multiturn is an expensive operation
        if process_multiturn_functions.call_all_processors(
            c[0], info[0]
        ):  # if programmatic validation passes
            return c

        retries += 1
        if retries >= 2:
            return None
        # If we're here, majority of relevance checks failed
        print("----------------\n\n\n\nRETRYING!!!!\n\n\n\n----------------")
        # Broken info is 1) rare and 2) handled by the retry limit. We don't want to waste compute on regenerating info as they take time.
        retry = await make_multiturn_conversation(info, multi_turn_conv_generator,completion_mode=completion_mode)
        if retry is not None:  # Note: retry CANNOT actually be None
            c = retry
        else:
            # If we failed to generate a retry, don't waste compute
            return None

    return None


async def make_multiturn_conversation(info, multi_turn_conv_generator, completion_mode=None):
    charname = extract_name.extract_name(info[1])
    conv_starter = create_conv_starter(info[1])
    if completion_mode:
        conv, conv_output = await multi_turn_conv_generator.generate(
            arguments = {
                "character": info[1].strip(),
                "scenario": info[2].strip(),
                "extra_info": extract_steps(info[3].strip()),
                "question_answer_list": format_qatuples(info[0]).strip(),
                "charname": charname.strip(),
                "conv_starter": conv_starter.strip(),
            }
        )
    else:
        conv, conv_output = await multi_turn_conv_generator.generate(
        arguments = {
            "character": info[1].strip(),
            "scenario": info[2].strip(),
            "extra_info": info[3].strip(),
            "question_answer_list": escape_unescaped_quotes(format_qatuples(info[0])).replace("\n","\\n"),
            "charname": charname.strip(),
            "conv_starter": conv_starter.strip(),
        }
        )
    output = get_output_directory()
    write_output_to_file(conv_output, output + "/multiturn_conversation_generations", info[4])

    return (conv, info[1], info[2], info[3], info[0])

def select_variation(character): # can help following the groove of the few-shot examples, in the case where you're using a slightly stupid model or low temperature
    charname=extract_name.extract_name(character)
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

    return random.choice(variations)

def fix_scenario_plan(scenario_plan, character):
    charname = extract_name.extract_name(character)
    if not ("Albert" in charname):
        if "Albert" in scenario_plan:
            print("Random Name was used instead of Albert")
        scenario_plan = scenario_plan.replace("Albert", random_name.random_name())
    return scenario_plan

def create_character_info_generators(completion_mode=None,engine_wrapper=None,logging_level=None,use_filenames=False):
    character_card_plan_path = "create_character_card_plan_no_filenames"
    if use_filenames:
        character_card_plan_path = "create_character_card_plan"
        
    character_card_plan_regex = re.compile(
        r"Character card plan \(be creative, do not use real people as characters, do NOT make the author of the book a character\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )

    try:
        overrides = LLM['override_character_card_plan_presets']

        if overrides['sampling_params'] is not None:
            sampling_params = overrides['sampling_params']

        if overrides['prompt'] is not None:
            character_card_plan_path = overrides['prompt']
    except KeyError:
        logger.info(f"Overrides for ''")

    if completion_mode:
        character_card_plan_path = character_card_plan_path + ".txt"
    else:
        character_card_plan_path = character_card_plan_path + ".json"

    character_card_plan_creator = GenerationStep(
        prompt_path=character_card_plan_path,
        regex = character_card_plan_regex,
        sampling_params={
        "max_tokens": 3000,
        "stop": [
            "### Response",
            "\n\n\n\n\n",
            "</s>",
            "# Input:",
            "[INST]",
            "### Instruction",
            "[INST",
            "## Character card plan (be creat",
            # "### Questions",
            "## Questions, answer, and text that the character should know:",
        ],
        "temperature": 1,
        # top_k=-1,
        "top_p": 0.5,
        # min_p=0.4,
        },
        completion_mode=completion_mode,
        logging_level=logging_level,
        retries=1,
        engine_wrapper=engine_wrapper,
        prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
    )
    
    # Character card gen
    
    character_card_path = "create_character_card_no_filenames"
    if use_filenames:
        character_card_path = "create_character_card"
        
    character_card_regex = re.compile(
        r"Character card \(be creative, write at least 3 paragraphs for each dialogue line\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    
    if completion_mode:
        character_card_path = character_card_path + ".txt"
    else:
        character_card_path = character_card_path + ".json"
    
    if obj_conf["SYSTEM"]["COMPLETION_MODE"]:
        stop_list = [
            "### Response",
            "\n\n\n\n\n",
            "</s>",
            "# Input:",
            "[INST]",
            "### Instruction",
            "[INST",
            "## Text",
            "## Character card",
        ]
    else:
        stop_list = [
            "### Response",
            "\n\n\n\n\n",
            "</s>",
            "# Input:",
            "[INST]",
            "### Instruction",
            "[INST",
            "## Text",
        ]
    
    character_card_creator = GenerationStep(
        prompt_path=character_card_path,
        regex = character_card_regex,
        sampling_params={
        "max_tokens": 4000,
        "stop": stop_list,
        "temperature": 1,
        "top_p": 0.5,
        },
        completion_mode=completion_mode,
        logging_level=logging_level,
        retries=1,
        engine_wrapper=engine_wrapper,
        prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
    )
    
    # Scenario Plan Gen
    scenario_plan_path = "create_scenario_plan" # no variation between use of filenames or not for scenarios
    
    scenario_plan_regex = re.compile(
        r"Scenario plan \(be creative, and make sure all characters present fit in with the setting\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    
    if completion_mode:
        scenario_plan_path = scenario_plan_path + ".txt"
    else:
        scenario_plan_path = scenario_plan_path + ".json"
    
    scenario_plan_creator = GenerationStep(
        prompt_path=scenario_plan_path,
        regex = scenario_plan_regex,
        sampling_params={
        "max_tokens": 8000,
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
            # "## Scenario",
        ],
        "temperature": 0.6,
        # top_k=-1,
        "top_p": 1,
        # min_p=0.5,
    },
        completion_mode=completion_mode,
        logging_level=logging_level,
        retries=1,
        engine_wrapper=engine_wrapper,
        prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
    )
    
    # Scenario Gen
    scenario_path = "create_scenario" # no variation between use of filenames or not for scenarios
    
    scenario_regex = re.compile(
        r"Scenario \(will have no dialogue, will just set up the scene\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    
    if completion_mode:
        scenario_path = scenario_path + ".txt"
    else:
        scenario_path = scenario_path + ".json"
    
    scenario_creator = GenerationStep( # will have variations as an argument
        prompt_path=scenario_path,
        regex = scenario_regex,
        sampling_params={
        "max_tokens": 8000,
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
            # "## Scenario",
        ],
        "temperature": 0.5,
        # top_k=-1,
        "top_p": 0.5,
        # min_p=0.5,
    },
        completion_mode=completion_mode,
        logging_level=logging_level,
        retries=1,
        engine_wrapper=engine_wrapper,
        prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPTS
    )
    
    return character_card_plan_creator, character_card_creator, scenario_plan_creator, scenario_creator


async def create_info(
    idx,
    group,
    engine_wrapper,
    assistant_mode,
    multi_turn_convs_info,
    multi_turn_convs_info_dir,
    rearrangements_to_take=3,
    use_filenames=False,
    completion_mode=None,
    logging_level=logging.INFO,
):
    # NOTE we set up all the generators up here so that we don't have to drill the args down like this is an old version of React
    # Instead we drill the generators down like it's an old version of React lol
    character_card_plan_creator, character_card_creator, scenario_plan_creator, scenario_creator = create_character_info_generators(
        engine_wrapper=engine_wrapper, use_filenames=use_filenames, completion_mode=completion_mode, logging_level=logging_level
    )
    
    # Resume normal control flow code
    all_permutations = list(itertools.permutations(group))

    sample_size = min(rearrangements_to_take, len(all_permutations))
    sampled_permutations = random.sample(all_permutations, sample_size)

    group_convs_info = []

    for iter, perm in enumerate(sampled_permutations):
        file_path = os.path.join(multi_turn_convs_info_dir, f"info_{idx}_{iter}.json")

        # Skip if file already exists
        if not os.path.exists(file_path):
            try:
                info = await make_multiturn_conversation_info(
                    perm, assistant_mode=assistant_mode, character_card_plan_creator=character_card_plan_creator, character_card_creator=character_card_creator, scenario_plan_creator=scenario_plan_creator, scenario_creator=scenario_creator,
                    completion_mode=completion_mode
                )

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
    idx, info, LLM, multi_turn_convs, multi_turn_convs_dir, assistant_mode=False, completion_mode=None, logging_level=logging.INFO
):
    file_path = os.path.join(multi_turn_convs_dir, f"conv_{idx}.json")
    multi_turn_conversation_prompt_path = "multi_turn_conversation"
    engine_wrapper = LLM['llm']

    if completion_mode:
        multi_turn_conversation_prompt_path = PROMPT_DICT['multi_turn_conversation']
    else:
        multi_turn_conversation_prompt_path = multi_turn_conversation_prompt_path + ".json"
        
    if assistant_mode:
        multi_turn_conversation_prompt_path = PROMPT_DICT['multi_turn_conversation_assistant_mode']

    sampling_params={
            "max_tokens": 8000,
            "stop": [
                "### Response",
                "\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "### Information",
                "## Information",
                "## Instruction",
                "Name:",
            ],
            "temperature": 0.8,
            # "top_k": -1,
            "top_p": 1,
            # "min_p": 0.6,
            }

    try:
        overrides = LLM['llm']
    except KeyError:
        logger.info("No overrides found for 'create_conversation' function. Using default settings.")
    try:
        sampling_params = overrides['sampling_params']
    except KeyError:
        logger.info("No sampling override found for 'create_conversation' function. Using default sampling settings.")
    try:
        prompt = overrides['prompt']
    except KeyError:
        logger.info("No prompt override found for 'create_conversation' function. Using default prompt.")

    qatuples = info[0]
    character = info[1]
    scenario = info[2]
    scenario_plan=info[3]
    
    charname = extract_name.extract_name(character)
    
    conversation_regex = re.compile(
        f"Conversation that answers the provided question \(be sure that you do not change the questions or answers themselves; {charname} will answer the questions, not ask them; the questions and answers provided should be copied word for word, and surrounded by compelling conversation\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    
    
    multi_turn_conv_generator = GenerationStep(
        prompt_path=multi_turn_conversation_prompt_path,
        regex=conversation_regex,
        sampling_params=sampling_params,
        completion_mode=completion_mode,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level,
        prompt_folder=folder_paths.get_prompts_directory(),
        default_prompt_folder=folder_paths.get_default_prompts_directory(),
    )

    # Skip if file already exists
    if not os.path.exists(file_path):
        try:
            conv = await make_multiturn_conversation(
                info, multi_turn_conv_generator, completion_mode=completion_mode
            )
            final_conv = await ensure_multiple_answers_are_same(
                info, conv, multi_turn_conv_generator, completion_mode=completion_mode
            )

            if final_conv is not None:
                if assistant_mode:
                    final_conv = (final_conv[0],"AI Assistant","A conversation between a helpful AI Assistant, and a user.", "N/A",final_conv[4])
                with open(file_path, "w") as file:
                    json.dump(final_conv, file, indent=4)

            multi_turn_convs.append(final_conv)
        except Exception as e:
            traceback.print_exc()
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
                    primary_char_name = extract_name.extract_name(primary_char_desc)
                    dialogues = process_multiturn_functions.extract_conversation(
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
    write_1 = obj_conf["PATH"]["OUTPUT"] + "/master_list.jsonl"
    with open(write_1, "w") as file:
        for item in master_list:
            file.write(json.dumps(item) + "\n")

    # Write the simplified data to a different .jsonl file
    write_2 = obj_conf["PATH"]["OUTPUT"] + "/simplified_data.jsonl"
    with open(write_2, "w") as file:
        for item in simplified_list:
            file.write(json.dumps(item) + "\n")

    print(
        f"Conversion complete. Master list written to {write_1}. Simplified data written to {write_2}."
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
                    conversations = process_multiturn_functions.extract_conversation(
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
    with open(obj_conf["PATH"]["OUTPUT"] + "/processed_master_list.json", "w") as file:
        json.dump(master_list, file)

    logger.info("Conversion complete. The processed master list is written to 'processed_master_list.json'.")



augmentoolkit.escape_unescaped_quotes


import aiohttp
import asyncio
import json

async def make_async_api_call(prompt=None, sampling_parameters={}, url='http://127.0.0.1:8080', messages=None):
    # Determine the endpoint based on the presence of messages
    if messages is not None:
        endpoint = "/v1/chat/completions"
        data = json.dumps({
            "messages": messages,
            **sampling_parameters  # Assuming sampling parameters can be applied to chat
        })
    else:
        endpoint = "/completion"
        data = json.dumps({
            "prompt": prompt,
            **sampling_parameters
        })

    # Complete the URL with the chosen endpoint
    full_url = url + endpoint

    # Use aiohttp to make the async request
    async with aiohttp.ClientSession() as session:
        async with session.post(full_url, data=data, headers={"Content-Type": "application/json"}, ssl=False) as response:
            if response.status == 200:
                # Parse the JSON response
                response_json = await response.json()
                if prompt:
                    return prompt + response_json["content"]
                else:
                    return response_json["choices"][0]["content"]
            else:
                return {"error": f"API call failed with status code: {response.status}"}




import re

# from .character_card_grammar import character_card_grammar
from .format_qatuples import format_qatuples
import string
import random


def extract_author_name(title):
    pattern = re.compile(r"\b(?:by|By)\s+([^,]+),")
    match = re.search(pattern, title)
    if match:
        author_name = match.group(1)
    else:
        author_name = [False]
    return author_name[0]  # first letter of Author name


augmentoolkit.select_random_capital

augmentoolkit.extract_capital_letters


import asyncio
import uuid
from openai import AsyncOpenAI
from augmentoolkit.generation_functions.async_llamacpp_api_call import make_async_api_call

try:
    from aphrodite import (
        EngineArgs,
        AphroditeEngine,
        SamplingParams,
        AsyncAphrodite,
        AsyncEngineArgs,
    )
except:
    print("Aphrodite not installed; stick to Llama CPP or API modes")

def make_id():
    return str(uuid.uuid4())


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

    async def submit_completion(
        self, prompt, sampling_params
    ):  # Submit request and wait for it to stream back fully
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

        if self.mode == "llamacpp": # Due to the way augmentoolkit was set up within Comfy, this path should never be called.
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
    
    async def submit_chat(
    self, messages, sampling_params
    ):  # Submit request and wait for it to stream back fully
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = []
        
        if self.mode == "llamacpp": # Due to the way augmentoolkit was set up within Comfy, this path should never be called.
            return await make_async_api_call(messages=messages, sampling_parameters=sampling_params)

        elif self.mode == "api":
            if DEBUG_MODE:
                logger.info(f"\n\n\nMESSAGES\n\n\n{messages}\n")

            messages_cleaned = [{"role": message["role"], "content": message["content"].replace("\\n","\n")} for message in messages]
            
            if DEBUG_MODE:
                logger.info(f"\n\n\nMESSAGES CLEANED\n\n\n{messages_cleaned}\n")

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

augmentoolkit.extract_name

augmentoolkit.extract_question_answer

augmentoolkit.format_qatuples

augmentoolkit.format_qatuples_noquotes

class GenerationStep:
    def __init__(self,
                 prompt_path="", # relative to the Inputs directory. Now just a string! - KR
                 regex=re.compile(r'.*', re.DOTALL), # take whole completion
                 sampling_params={ # Can be overriden with LLM['sampling_params']
                     "temperature": 1,
                     "top_p": 1,
                     "max_tokens": 3000,
                     "stop": [
                            "### Response",
                            "\n\n\n\n\n",
                            "</s>",
                            "# Input:",
                            "[INST]",
                            "### Instruction",
                            "### Information",
                            "## Information",
                            "## Instruction",
                            "Name:",
                        ]
                    },
                 completion_mode=True, # Chat vs completion mode
                 retries=0,
                 engine_wrapper=None, # This is the LLM dictionary object LLM['llm']
                 logging_level=logging.INFO,  # Default logging level
                 output_processor=lambda x: x, # to ensure that control flow does not need to have decision code handling the outputs of the LLM, you can pass in a function to handle and modify the outputs (post regex) here. By default it's just the identity function and does nothing.
                 return_input_too=True, 
                 default_prompt_folder="prompts",
                 prompt_folder="prompts"
                ):
        self.prompt_path = prompt_path
        self.regex = regex
        self.sampling_params = sampling_params
        self.completion_mode = completion_mode
        self.retries = retries
        self.logging_level = logging_level
        self.output_processor = output_processor
        self.return_input_too = return_input_too
        if not engine_wrapper:
            raise Exception("Engine wrapper not passed in!")
        self.engine_wrapper = engine_wrapper 
        self.prompt_folder = prompt_folder
        self.default_prompt_folder = default_prompt_folder
        logging.basicConfig(level=self.logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

    
    async def generate(self,arguments={}):

        try:
            # Current file directory
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Dynamic INPUT_DIRECTORY path
            ideal_path = os.path.join(current_dir, '..', '..', self.prompt_folder,self.prompt_path)
            if os.path.exists(ideal_path):
                full_prompt_path = ideal_path
            else:
                full_prompt_path = os.path.join(current_dir, '..', '..', self.default_prompt_folder,self.prompt_path)

            if full_prompt_path.endswith(".json"): #JSON route. This is a slightly-modified version of augmentoolkit's original prompt import code.
                with open(full_prompt_path, 'r') as pf:
                    prompt = pf.read()
                    # Code to ensure that interpolation works, but curly braces are still allowed in the input
                    # 1. Escape all curly braces
                    prompt_escaped = prompt.replace('{', '{{').replace('}', '}}')
                    # 2. Unescape curly braces that are associated with input keys
                    for key in arguments.keys():
                        prompt_escaped = prompt_escaped.replace(f"{{{{{key}}}}}", f"{{{key}}}") # Somehow this works
                    # 3. Format
                    prompt_formatted = prompt_escaped.format(**arguments)
            else: # The PROMPT_DICT route.
                prompt_name = os.path.splitext(os.path.basename(full_prompt_path))[0] # Extract the prompt name from its file path.
                prompt_formatted, _ = load_external_prompt_and_grammar(prompt_name, "dummy_grammar", arguments) # Load the prompt with its arguments filled in.

        except Exception as e:
            logger.exception(f"An exception occured when trying to load the prompt '{self.prompt_path}': {e}")

        times_tried = 0
        if self.completion_mode:
            while times_tried <= self.retries:
                try:
                    response = await self.engine_wrapper['llm'].submit_completion(prompt_formatted, self.sampling_params)
                    filtered_response = re.search(self.regex, response).group(1)
                    ret = self.output_processor(filtered_response)
                    if self.return_input_too:
                        return ret, prompt_formatted + filtered_response
                    return ret
                except Exception as e:
                    if DEBUG_MODE:
                        logger.error(f"Error in Generation Step: {e}")
                    try:
                        if not self.engine_wrapper['llm'].mode == "llamacpp":
                            print("Response:")
                            print(response)
                    except:
                        pass
                    traceback.print_exc()
                    times_tried += 1
            raise Exception("Generation step failed -- too many retries!")
        else:
            while times_tried <= self.retries:
                try:
                    messages = json.loads(prompt_formatted)
                    response = await self.engine_wrapper['llm'].submit_chat(messages, self.sampling_params)
                    filtered_response = response.replace('"','\\"').replace("\n","\\n")
                    ret = self.output_processor(filtered_response)
                    if self.return_input_too:
                        return ret, json.dumps(messages + [{"role": "assistant", "content": filtered_response}])
                    return ret
                except Exception as e:
                    logger.error(f"Error in Generation Step: {e}")
                    logger.info(prompt_formatted)
                    logger.error(f"Above prompt resulted in error, probably the model's fault: {e}")
                    traceback.print_exc()
                    times_tried += 1
            raise Exception("Generation step failed -- too many retries!")




























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
            logger.info(f'\n\nSETTINGS DUMP\n\nmodel:{self.model}\nprompt:{prompt}\ntemperature:{sampling_params["temperature"]}\ntop_p:{sampling_params["top_p"]}\nmax_tokens:{sampling_params["max_tokens"]}\n')

        if self.mode == "llamacpp": # Should never reach this route, as mode will never be set to this value.
            return await make_async_api_call(prompt=prompt, sampling_parameters=sampling_params)
        
        if self.mode == "aphrodite":
            aphrodite_sampling_params = SamplingParams(**sampling_params)
            request_id = make_id()
            outputs = []
            final_output = None

            async for request_output in self.engine.generate(
                prompt, aphrodite_sampling_params, request_id
            ):
                outputs.append(request_output.outputs[0].text)
                final_output = request_output

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
    
    async def submit_chat(self, messages, sampling_params): 
        # Submit request and wait for it to stream back fully
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
            
            if DEBUG_MODE: # Input messages
                logger.info(f"\nMESSAGES\n{messages}\n")

            messages_cleaned = [{"role": message["role"], "content": message["content"].replace("\\n","\n")} for message in messages]

            if DEBUG_MODE: # Output messages
                logger.info(f"messages_cleaned\n{messages_cleaned}\n")

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





































# This is hard-coded into the node, but for async purposes, we'll keep this copy here.
def identify_duplicates(
    tuples: List[Tuple[str, str, str, str]]
) -> List[Tuple[str, str, str, str]]:
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

augmentoolkit.has_sequential_chars

augmentoolkit.extract_conversation

augmentoolkit.compare_answers_with_qatuples

augmentoolkit.check_for_repeated_dialogue_answers

augmentoolkit.check_conversation_length

augmentoolkit.check_conversation_for_text_from_examples

augmentoolkit.check_each_question_contains_q_from_tuples

augmentoolkit.check_for_unintended_repeated_quotes

augmentoolkit.call_all_processors

augmentoolkit.random_name

augmentoolkit.strip_steps


import yaml
import os
import uuid
# This is in no way best practices, but all my prompts being searchable and separate files is a good way to make my life easier.
import pkgutil
import importlib
import sys
from tqdm import asyncio as tqdmasyncio
import asyncio
import json
import os
from transformers import AutoTokenizer
import re
from tqdm import tqdm
import nltk
import glob
import os

def convert_to_utf8(file_path):
    with open(file_path, 'r', encoding='latin-1') as f:
        content = f.read()

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def convert_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f'Converting {file_path} to UTF-8...')
                convert_to_utf8(file_path)


directory = folder_path
convert_files_in_directory(directory)


# We have to define this up here so that two-step generation works, you'll see later.
multi_turn_convs_info_dir = OUTPUT + "/multi_turn_convs_info"  # we generate all the information fed to the multiturn prompt, and generate the actual multiturn prompt, separately; since every step but the last is capable of being done by a 13b



# First, import all modules so they can be reloaded
for _, module_name, _ in pkgutil.iter_modules(
    generation_functions.__path__, generation_functions.__name__ + "."
):
    importlib.import_module(module_name)

# TODO This might not be necessary for augmentoolkit in Comfy. Check to see if it is.
# Now, reload each module and import all callable attributes
for _, module_name, _ in pkgutil.iter_modules(
    generation_functions.__path__, generation_functions.__name__ + "."
):
    # Reload the module
    module = importlib.reload(sys.modules[module_name])
    # Iterate through each attribute in the reloaded module
    for attribute_name in dir(module):
        # Retrieve the attribute
        attribute = getattr(module, attribute_name)
        if callable(attribute):
            # If it's callable, it's a function or class, so you set it in the globals dictionary
            globals()[attribute_name] = attribute



# Revise QA tuples node
# Directory for QA tuples
qa_tuples_dir = OUTPUT + "/qatuples_raw"
if not os.path.exists(qa_tuples_dir):
    os.makedirs(qa_tuples_dir)

# Initialize vetted_qa_tuples
vetted_qa_tuples = []  # tuple list of qa tuples that have been judged good

# Attempt to initialize filtered_worthy_for_questions
try:
    _ = filtered_worthy_for_questions
except NameError:
    filtered_worthy_for_questions = []

if not filtered_worthy_for_questions:
    # Load all files in the qa_tuples_dir if filtered_worthy_for_questions is not initialized
    existing_files = glob.glob(os.path.join(qa_tuples_dir, "*.json"))
    for file_path in existing_files:
        with open(file_path, "r") as file:
            qa_tuple = tuple(json.load(file))
            print(f"Loaded {file}")
        vetted_qa_tuples.append(qa_tuple)
else:
    tasks = [control_flow_functions.generate_qatuples_from_para(
        idx,
        para,
        engine_wrapper=engine_wrapper,
        vetted_qa_tuples=vetted_qa_tuples,
        qa_tuples_dir=qa_tuples_dir,
        double_check_counter=obj_conf['SYSTEM']['DOUBLE_CHECK_COUNTER'],
        use_filenames=obj_conf['SYSTEM']['USE_FILE_NAMES']) for idx,para in enumerate(filtered_worthy_for_questions)]
    limited_tasks_qgen = [run_task_with_limit(task) for task in tasks]
    
    async def run_tasks(limited_tasks_qgen):
        for future in tqdmasyncio.tqdm.as_completed(limited_tasks_qgen):
            await future

    asyncio.run(run_tasks(limited_tasks_qgen))


# Print stats related to revised qatuples, and filter out nones (questions that were unanswerable due to lack of context).
import json
import os

print("-------------- QUESTIONS REVISED ------------- STATS SO FAR:")
nones = list(filter(lambda x: x is None, vetted_qa_tuples))
print(f"Nones: {len(nones)}")
print(f"Non-nones: {len(vetted_qa_tuples) - len(nones)}")
print(f"Total: {len(vetted_qa_tuples)}")
# filter out all None values
vetted_qa_tuples = [qa for qa in vetted_qa_tuples if qa is not None]
print("---------------- ONTO EXAMPLES GENERATION-------------------")


# Group by text node
qa_tuples_by_paragraph = control_flow_functions.group_by_text(vetted_qa_tuples)



if not os.path.exists(multi_turn_convs_info_dir):
    os.makedirs(multi_turn_convs_info_dir)


# In[ ]:

# Multiturn Conversation Node
import json
import random
import itertools

multi_turn_convs_info = []


tasks = [control_flow_functions.create_info(idx,group,engine_wrapper, obj_conf['SYSTEM']['ASSISTANT_MODE'], multi_turn_convs_info,multi_turn_convs_info_dir, obj_conf['SYSTEM']['REARRANGEMENTS_TO_TAKE'],obj_conf['SYSTEM']['USE_FILE_NAMES']) for idx,group in enumerate(qa_tuples_by_paragraph)]

async def limited_tasks():
   
    limited_tasks_infocreation = [run_task_with_limit(task) for task in tasks]

    async def process_tasks(future, as_completed, limited_tasks_infocreation):
        for future in tqdmasyncio.tqdm(as_completed(limited_tasks_infocreation)):
            await future

    # Run the async function
    await process_tasks()

# Call the main async function
asyncio.run(limited_tasks())



# Make Dataset node
import os
import json
import random
import itertools
import asyncio

multi_turn_convs_dir = config["PATH"]["OUTPUT"] + "/multi_turn_convs"
if not os.path.exists(multi_turn_convs_dir):
    os.makedirs(multi_turn_convs_dir)

multi_turn_convs = []

tasks = [control_flow_functions.create_conversation(
    idx,info, engine_wrapper, 
    multi_turn_convs, 
    multi_turn_convs_dir, 
    assistant_mode=ASSISTANT_MODE, 
    completion_mode=COMPLETION_MODE, 
    logging_level=LOG_LEVEL) for idx,info in enumerate(convs_info)
]
limited_tasks_convwriting = [run_task_with_limit(task) for task in tasks]
for future in tqdmasyncio.tqdm.as_completed(limited_tasks_convwriting):
    await future


