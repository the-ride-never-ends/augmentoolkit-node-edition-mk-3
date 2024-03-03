import asyncio
import inspect
import json
import logging
import os
import random
import re
import sys
import traceback
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from collections import Counter
from datetime import datetime
from llama_cpp import Llama, LlamaGrammar
from math import ceil
from PIL import Image, ImageOps, ImageSequence
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))
custom_nodes_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ComfyUI", "custom_nodes")
sys.path.insert(0, custom_nodes_path)

from custom_nodes import helper_functions, grammars, logger
from custom_nodes.helper_functions import format_external_text_like_f_string, make_id, load_external_prompt_and_grammar
from custom_nodes.logger import logger

import folder_paths

#########################################
#### PROMPT AND GRAMMAR DICTIONARIES ####
#########################################

# Create dictionaries for the prompts and grammars from the txt files in the prompts and grammars folders. 
# The prompt names from the dictionary must match the name of the function they go to + "_prompt".

# Create an empty dictionary object to put the prompts into.
PROMPT_DICT = {}

# For every prompt txt file in the prompts folder...
for file_name in folder_paths.get_filename_list("prompts"):
    try:
        # Get the full file path for prompt txt file.
        file_path = folder_paths.get_full_path("prompts", file_name)

        # Open the file for the prompt and put it in the prompt dictionary
        with open(file_path, 'r', encoding='utf-8') as file:
            key = os.path.splitext(file_name)[0]
            PROMPT_DICT[key] = file.read()

    except Exception as e:
        logger.exception(f"An Exception occured when creating the prompt dictionary object: {e} ")

# Create an empty dictionary object to put the grammars into.
GRAMMAR_DICT = {}

# For every grammar txt file in the grammars folder...
for file_name in folder_paths.get_filename_list("grammars"):
    try:
        # Get the full file path for the grammar txt file.
        file_path = folder_paths.get_full_path("grammars", file_name)

        # Open the file for the grammar and put it in the grammar dictionary.
        with open(file_path, 'r', encoding='utf-8') as file:
            key = os.path.splitext(file_name)[0]
            GRAMMAR_DICT[key] = file.read()

    except Exception as e:
        logger.exception(f"An Exception occured when creating the grammar dictionary object: {e} ")

##############################
#### INDIVIDUAL FUNCTIONS ####
##############################

# all characters in this prompt are over 18

# Explanation of wtf the first few-shot example is:
# No I do not have a teacher-student fetish, the reason why Elise is a teacher is an adaptation to the following three facts:
# 1. This tool is meant to be able to generate data for training ERP bots by default
# 2. This tool is also meant to be able to take in educational material by default
# 3. When generating characters that would know about educational material, the model tends to generate academics or professors in that field, talking to students.
# Given these facts, we clearly need to prompt the model to be able to generate horny teachers, or else it's going to just do it poorly when it realizes it has a sexualized character that's also a teacher. I didn't want to choose this, the constraints of the problem forced me to.

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

######################
#### NODE CLASSES ####
######################

class ArmchairPsychologist: # Add in support for off-loaded prompting.
    """
    This function grades an LLM's adherence to an assigned personality trait.
    It then saves the information and the synthetic data it generates to a json called 'adhere.json'.
    :param: character_card: A character card string.
    :param: rp_initialized_model: An initialized model, with parameters set for role-play. This model will "play" the character.
    :param: initialized_model: An initialized model, with parameters set for logic and determinism. This model will "play" the psychologist.
    :param: override_llm_presets: Override the llm's presets 
    :param: total_retries: Total number of times the test should be conducted. Higher generates greater statistical certainty but longer run-time.
    :param: big_five_trait: Whether the LLM is measuring a Big Five personality trait. 
    :return: adherence: An average score of how reliably the character embodied their assigned personality trait.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "character_card": ("OUTPUT_TEXT",),
                "rp_initialized_model": ("INITIALIZED_MODEL",),
                "initialized_model": ("INITIALIZED_MODEL",),
                "override_llm_presets": ("OVERRIDE_LLM_PRESETS_CHOICE",),
                "total_retries": ("INT", {"default": 5, "min": 1, "max": 16641, "step":1}), # Max is the total sample size necessary for a 99% confidence level, +/- 1%, for an unknown (50%) population proportion.
                "big_five_trait": (["True", "False"],),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "armchair_psychologist"

    CATEGORY = "output_validation"

    def armchair_psychologist(character_card: str, total_retries:int, initialized_model, override_llm_presets, big_five_trait: str) -> None:
        # Initialize counters and other variables
        start_time = time.time()
        retries = 0
        score = 0
        char_response_dic = {}
        loop_dic = {}

        # Initialize grammar for the psychologist judgment prompt
        psychologist_judgement_prompt_grammar = LlamaGrammar.from_string(
        r"""
        root ::= reasoning answer
        reasoning ::= 
        answer ::= ("Very Low" | "Moderately Low" | "Neutral" | "Moderately High" | "Very High")
        """
        )

        yes_no_grammar = LlamaGrammar.from_string(
        r"""
        root ::= answer
        answer ::= ("Yes" | "No")
        """
        )

        # Define the 10 Big Five personality trait subtypes
        personality_description = [
            "High Openness to Experience. Openness to experience refers to one’s willingness to try new things as well as engage in imaginative and intellectual activities. It includes the ability to “think outside of the box.” Those high in openness to experience are perceived as creative and artistic. They prefer variety and value independence. They are curious about their surroundings and enjoy traveling and learning new things.",
            "Low Openness to Experience. Openness to experience refers to one’s willingness to try new things as well as engage in imaginative and intellectual activities. It includes the ability to “think outside of the box.” Those low in openness to experience prefer routine. They are uncomfortable with change and trying new things, so they prefer the familiar over the unknown. As they are practical people, they often find it difficult to think creatively or abstractly.",
            "High Conscientiousness. Conscientiousness describes a person’s ability to regulate impulse control in order to engage in goal-directed behaviors. It measures elements such as control, inhibition, and persistence of behavior. Those high in conscientiousness can be described as organized, disciplined, detail-oriented, thoughtful, and careful. They also have good impulse control, which allows them to complete tasks and achieve goals.",
            "Low Conscientiousness. Conscientiousness describes a person’s ability to regulate impulse control in order to engage in goal-directed behaviors. It measures elements such as control, inhibition, and persistence of behavior. Those low in conscientiousness may struggle with impulse control, leading to difficulty in completing tasks and fulfilling goals. They tend to be more disorganized and may dislike too much structure. They may also engage in more impulsive and careless behavior.",
            "High Extraversion. Extraversion reflects the tendency and intensity to which someone seeks interaction with their environment, particularly socially. It encompasses the comfort and assertiveness levels of people in social situations. Those high in extraversion are generally assertive, sociable, fun-loving, and outgoing. They thrive in social situations and feel comfortable voicing their opinions. They tend to gain energy and become excited from being around others.",
            "Low Extraversion. Extraversion reflects the tendency and intensity to which someone seeks interaction with their environment, particularly socially. It encompasses the comfort and assertiveness levels of people in social situations. Those low in extraversion are often referred to as introverts. These people tend to be more reserved and quieter. They prefer listening to others rather than needing to be heard. Introverts often need periods of solitude in order to regain energy as attending social events can be very tiring for them. Of importance to note is that introverts do not necessarily dislike social events, but instead find them tiring.",
            "High Agreeableness. Agreeableness refers to how people tend to treat relationships with others, and focuses on people’s orientation and interactions with others. Those high in agreeableness can be described as soft-hearted, trusting, and well-liked. They are sensitive to the needs of others and are helpful and cooperative. People regard them as trustworthy and altruistic.",
            "Low Agreeableness. Agreeableness refers to how people tend to treat relationships with others, and focuses on people’s orientation and interactions with others. Those low in agreeableness may be perceived as suspicious, manipulative, and uncooperative. They may be antagonistic when interacting with others, making them less likely to be well-liked and trusted.",
            "High Neuroticism. Neuroticism describes the overall emotional stability of an individual through how they perceive the world. It takes into account how likely a person is to interpret events as threatening or difficult. It also includes one’s propensity to experience negative emotions. Those high in neuroticism often feel anxious, insecure and self-pitying. They are often perceived as moody and irritable. They are prone to excessive sadness and low self-esteem.",
            "Low Neuroticism. Neuroticism describes the overall emotional stability of an individual through how they perceive the world. It takes into account how likely a person is to interpret events as threatening or difficult. It also includes one’s propensity to experience negative emotions. Those low in neuroticism are more likely to calm, secure and self-satisfied. They are less likely to be perceived as anxious or moody. They are more likely to have high self-esteem and remain resilient."
        ]

        # Construct the path to situational judgment test json "jst.json" and the adherence output json "adhere.json". 
        # This assumes that the jsons are in the same file folder as the script.py
        script_dir = os.path.dirname(os.path.realpath(__file__))
        jst_file_path = os.path.join(script_dir, 'sjt.json')
        adhere_file_path = os.path.join(script_dir, 'adhere.json')

        # Load the jst dictionary
        try:
            with open(jst_file_path, 'r') as file:
                jst = json.load(file)
        except (json.decoder.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f'Error in loading jst dictionary json: {e}')

        # Load the adhere dictionary
        try:
            with open(adhere_file_path, 'r') as file:
                adhere = json.load(file)
        except (json.decoder.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f'Error in loading adhere dictionary json: {e}')

        adhere["char_index"] += 1
        char_key = f'char_index: {adhere["char_index"]}'

        adhere[char_key] = {
            'character_card': character_card,
            'tests': {}
        }

        while retries < total_retries: 
            if big_five_trait == "True":

                #Define the Big Five personality trait descriptions.
                #personality_description = [
                    #"Openness to Experience: Openness to Experience emphasizes imagination and insight the most out of all five personality traits. People who are high in openness to experience tend to have a broad range of interests. They are curious about the world and other people and are eager to learn new things and enjoy new experiences. People who are high in this personality trait also tend to be more adventurous and creative. Conversely, people low in this personality trait are often much more traditional and may struggle with abstract thinking.",
                    #"Conscientiousness: Conscientiousness is one defined by high levels of thoughtfulness, good impulse control, and goal-directed behaviors. Highly conscientious people tend to be organized and mindful of details. They plan ahead, think about how their behavior affects others, and are mindful of deadlines. Someone scoring lower in this primary personality trait is less structured and less organized. They may procrastinate to get things done, sometimes missing deadlines completely.",
                    #"Extraversion: Extraversion (or extroversion) is a personality trait characterized by excitability, sociability, talkativeness, assertiveness, and high amounts of emotional expressiveness. People high in extraversion are outgoing and tend to gain energy in social situations. Being around others helps them feel energized and excited. People who are low in this personality trait or introverted tend to be more reserved. They have less energy to expend in social settings and social events can feel draining. Introverts often require a period of solitude and quiet in order to “recharge.”",
                    #"Agreeableness: Agreeableness includes attributes such as trust, altruism, kindness, affection, and other prosocial behaviors. People who are high in agreeableness tend to be more cooperative while those low in this personality trait tend to be more competitive and sometimes even manipulative.",
                    #"Neuroticism. Neuroticism is a personality trait characterized by sadness, moodiness, and emotional instability. Individuals who are high in neuroticism tend to experience mood swings, anxiety, irritability, and sadness. Those low in this personality trait tend to be more stable and emotionally resilient."
                #]
                # Extract the character's assigned Big Five personality trait from their character card.
                regex_big_five = "\b(High|Low) (Openness to Experience|Conscientiousness|Extraversion|Agreeableness|Neuroticism)\b"
                assigned_trait = re.search(regex_big_five, character_card, re.IGNORECASE)

                if "\bHigh Openness to Experience\b" in assigned_trait:
                    personality_description = personality_description[0]

                elif "\bLow Openness to Experience\b" in assigned_trait:
                    personality_description = personality_description[1]

                elif "\bHigh Conscientiousness\b" in assigned_trait:
                    personality_description = personality_description[2]

                elif "\bLow Conscientiousness\b" in assigned_trait:
                   personality_description = personality_description[3]

                elif "\bHigh Extraversion\b" in assigned_trait:
                    personality_description = personality_description[4]

                elif "\bLow Extraversion\b" in assigned_trait:
                    personality_description = personality_description[5]

                elif "\bHigh Agreeableness\b" in assigned_trait:
                    personality_description = personality_description[6]

                elif "\bLow Agreeableness\b" in assigned_trait:
                    personality_description = personality_description[7]

                elif "\bHigh Neuroticism\b" in assigned_trait:
                    personality_description = personality_description[8]

                elif "\bLow Neuroticism\b" in assigned_trait:
                    personality_description = personality_description[9]

                else:
                    logger.warning("Warning: Regex of Big Five personality traits for the input character card did not find any assigned traits.")
                    break

                logger.info(f"Character's assigned Big Five personality trait: {personality_description}")

                # Make the character answer a situational judgment test question relevant to their given Big Five Personality trait.
                jst_trait_question = jst['trait']
                logger.info(f"Pulled JST trait question: {jst_trait_question}")
                character_jst_prompt = f"""
                {character_card}
                #######	
                Answer the following question. Your answer must match your personality descripition under special instructions.
                Question: {jst_trait_question}
                Answer:
                """

                # Save the character's answer.
                char_response = initialized_model(
                    character_jst_prompt,
                    grammar=None,
                )
                logger.info(f"Character's response to pulled JST trait question: {char_response}")

                # Reformat the results of the situational judgment test before giving it to the psychologist.
                character_jst_pattern = r".*?Answer the following question\. Your answer must match your personality description under special instructions\."
                character_jst_cut_down = re.sub(character_jst_pattern, "", character_jst_prompt, flags=re.DOTALL)

                text = character_jst_cut_down + char_response
                logger.info(f"Complete text: {text}")

                # Evaluate the character's answer using a customized variant of Prompt 3 from Tianlong et al. 2023
                psychologist_judgement_prompt = f"""
                You are an expert psychologist specializing in personality testing using the Big Five personality theory. 
                In this theory, there are five personality traits, each with two distinct sub-types, one of which is {personality_description} 
                The following text is a dialogue. Based on this dialogue, please judge how much the respondent reflects this personality sub-type. 
                Your answer can be only one of these five responses:
                Very Low, Moderately Low, Neutral, Moderately High, and Very High.
                #######
                Text: {text}
                Your Professional Judgement:
                """

                # Get the psychologist's evaluation.
                answer = initialized_model(
                    psychologist_judgement_prompt,
                    grammar=psychologist_judgement_prompt_grammar,
                )
                logger.info(f"Psychologist's judgement of Character's response to JST trait question: {answer}")

                # Turn his response into a numbered grade and add that onto the score counter
                if answer == "Very Low" or answer.lower() == "very low":
                    score += 1
                elif answer == "Moderately Low" or answer.lower() == "moderately low":
                    score += 2
                elif answer == "Neutral" or answer.lower() == "neutral":
                    score += 3
                elif answer == "Moderately High" or answer.lower() == "moderately high":
                    score += 4
                elif answer == "Very High" or answer.lower() == "very high":
                    score += 5

                # Save the results of all this to the json-like structure.
                test_key = f'test: {retries}'
                adhere[char_key]['tests'][test_key] = {
                    'test_num': test_count,
                    'text_score': score,
                    'assigned_trait': assigned_trait,
                    'jst_question': jst_trait_question,
                    'char_response': char_response
                }
                retries += 1  

            else: # This is where other personality trait schemes would go. Will probably be replaced by an elif since there are a LOT of personality schemas.
                pass

        # Calculate the adherence score based on the sum of all answers divided by the total possible score
        adherence = (score/(5*total_retries))
        logger.info(f"Overall Adherence of Character to Assigned Personality Trait: \n{adherence} ({adherence*100}%).\nSaving responses and their respective adherence weight.")

        # Save the adherence score to the json-like structure.
        adhere[char_key] = {'adherence': adherence}

        # Save the all the results of this function to the adhere.json
        with open(adhere_file_path, 'w') as file:
            json.dump(adhere, file, indent=4)

        end_time = time.time()
        elapsed_time = (end_time - start_time)
        logger.info(f"Adherence score calculated with {retries} in {elapsed_time} seconds.")
        return (adherence, { "ui": { "text": adherence } })

class EnsureMultipleAnswersAreSame: # This needs to be it's own function instead of a node. Bleh...
    """
    This function takes a json file (?) and determines which paragraphs are worth of making questions from.

    :param cleaned_paragraphs: chunked paragraphs from either the chunking function or a pre-made json
    :param model: chosen llm
    :param text_path: The file path to the chunked paragraphs
    :param tokenizer: tokenizer (original: SentencePiece)
    :param max_token_length: The maximum token length for a chunk of sentences
    :return: List of sentence chunks with source text information
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cleaned_text": ("OUTPUT_TEXT",),
                "initialized_model": ("INITIALIZED_MODEL",),
                "decision_prompt_judge_paragraphs_prompt": ("PROMPT",),
                "judge_paragraph_grammar": ("GRAMMAR",),
                #"test_judge_paragraphs": ("PROMPT",), # These are for debugging purposes, and probably don't need to be here.
                #"test2_judge_paragraphs": ("PROMPT",),
                #"test3_judge_paragraphs": ("PROMPT",),
                "text_manually_cleaned": (["True", "False"],),
                "max_token_length": ("INT", {"default": 400, "min": 1, "max": 10000, "step":1}),
                # override_llm_presets
            },
        }

    RETURN_TYPES = ("OUTPUT_TEXT",)
    FUNCTION = "return_judged_worthy_for_questions"

    CATEGORY = "output_validation"

    def ensure_multiple_answers_are_same(info, conv, scenario, initialized_model):  # why is this a whole separate function? Once upon a time, LLMs were used in validation here, too. But programmatic validation SEEMS to catch the common problems. This is here so that I can add it back in if I have to.
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
            logger.warning("----------------\n\n\n\nRETRYING!!!!\n\n\n\n----------------")
            # Broken info is 1) rare and 2) handled by the retry limit. We don't want to waste compute on regenerating info as they take time.
            retry = make_multiturn_conversation(info, initialized_model)
            if retry is not None:  # Note: retry CANNOT actually be None
                c = retry
            else:
                # If we failed to generate a retry, don't waste compute
                logger.warning("Failed to generate a retry in . Returning None and continuing.")
                return None

        return None

class FilterAndGraph: #TODO Write documentation for this function.
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
                "judged_worthy_for_questions": ("OUTPUT_TEXT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("filtered_worthy_for_questions",)
    FUNCTION = "filter_and_graph"

    CATEGORY = "output_validation"
    
    def filter_and_graph(self, judged_worthy_for_questions):

        filtered_worthy_for_questions = None

        try:
            # Count the occurrences of None and non-None for each source text.
            source_counts = Counter()
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
            logger.info(f"filtered_worthy_for_questions: {filtered_worthy_for_questions} : filtered_worthy_for_questions")

        except Exception as e:
            logger.exception(f"An Exception occurred in filter_and_graph function under class FilterAndGraph: {e}")

        return (filtered_worthy_for_questions, { "ui": { "paragraphs_suitability_plot": paragraphs_suitability_plot } })

class JudgeParagraphs: #TODO Refactor to move prompt and grammar to inside the node.
    """
    This function takes a json file (?) and determines which paragraphs are worth of making questions from.

    :param cleaned_text: chunked paragraphs from either the chunking function or a pre-made json (external).
    :param initialized_model: Chosen initialized llm (external).
    :param overide_llm_presets: Override the function's default llm presets (external).
    :param text_manually_cleaned: Load the chunked paragraphs from a pre-made json.
    :param max_token_length: The maximum token length for a chunk of sentences
    :return judged_worthy_for_questions: List of paragraphs deemed worthy for asking questions about.
    """

    @staticmethod
    def judge_paragraph(p, initialized_model, overide_llm_presets):
        reached_decision = False
        max_retries = 0
        prompt_content = {
            "p": p,
        }

        logger.info(f"\nParagraph being judged: \n{p} \nParagraph being judged \ntype: {type(p)}")
        time.sleep(10)

        while not reached_decision and (max_retries <= 3):

            # Load the prompt and the grammar.
            try:
                decision_prompt, judge_paragraph_grammar = load_external_prompt_and_grammar(inspect.currentframe().f_code.co_name,"judge_paragraph_grammar", prompt_content)
            except Exception as e:
                logger.exception(f"An Exception occured in 'judge_paragraph' function in class JudgeParagraphs: {e}")
                break

            # Load the initialized LLM and judge the paragraph.
            try:
                start_time = time.time()

                if overide_llm_presets:
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
                logger.info(f"\n*** Response for 'judge_paragraph' function ***\n{response}\n*** Response for 'judge_paragraph' function ***\n")

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

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cleaned_paragraphs": ("TUPLE", {"forceInput": True}),
                "initialized_model": ("INITIALIZED_MODEL",),
                "overide_llm_presets": ("OVERRIDE_LLM_PRESETS_CHOICE",),
                "text_manually_cleaned_arg": (["False", "True"],),
                #"max_token_length": ("INT", {"default": 400, "min": 1, "max": 10000, "step":1}),
            },
        }

    RETURN_TYPES = ("OUTPUT_TEXT",)
    RETURN_NAMES = ("judged_worthy_for_questions",)
    FUNCTION = "return_judged_worthy_for_questions"

    CATEGORY = "output_validation"

    def return_judged_worthy_for_questions(self, cleaned_paragraphs, initialized_model, overide_llm_presets, text_manually_cleaned_arg,):

        # Strip out the sources part of the tuple, leaving only the paragraphs.
        #cleaned_paragraphs = cleaned_paragraphs[0]

        worthy_for_questions_output_dir = "./worthy_for_questions"

        if not os.path.exists(worthy_for_questions_output_dir):
            os.makedirs(worthy_for_questions_output_dir, exist_ok=True)

        if text_manually_cleaned_arg == "False":
            # Create the question list
            judged_worthy_for_questions = []

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

                    # If the data is a string, don't do anything. If it isn't, append the paragraph and metadata.
                    if isinstance(data, str):
                        judged_worthy_for_questions.append((None, data[7:]))
                    else:
                        judged_worthy_for_questions.append((data["paragraph"], data["metadata"]))
                else:
                    judgment = self.judge_paragraph(paragraph, initialized_model, overide_llm_presets)
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

            return(judged_worthy_for_questions,)

        else:
            # No need to write to file since paragraph chunking is deterministic
            judged_worthy_for_questions = paragraphs_processed

            return(judged_worthy_for_questions,)


NODE_CLASS_MAPPINGS = {
    #Output Validation
    "ArmchairPsychologist": ArmchairPsychologist,
    #"CheckAnswer": CheckAnswer,
    #"CheckAnsewerRelevancyWithText": CheckAnsewerRelevancyWithText,
    #"CheckQuestion": CheckQuestion,
    #"CheckQATupleContext": CheckQATupleContext,
    "EnsureMultipleAnswersAreSame": EnsureMultipleAnswersAreSame,
    "FilterAndGraph": FilterAndGraph,
    #"IdentifyDuplicates": IdentifyDuplicates,
    "JudgeParagraphs": JudgeParagraphs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    #Output Validation
    "ArmchairPsychologist": "Check Character Card Personality Adherence",
    #"CheckAnswer": "Check Answer",
    #"CheckAnsewerRelevancyWithText": "Check Answer Relevancy with Text",
    #"CheckQuestion": "Check Question",
    #"CheckQATupleContext": "Check QA Tuple Context",
    "EnsureMultipleAnswersAreSame": "Ensure Multiple Answers are the Same",
    "FilterAndGraph": "Filter out and Graph 'None's from Judged Paragraphs",
    #"IdentifyDuplicates": "Identify Duplicates"
    "JudgeParagraphs": "Judge Paragraphs for QA Suitability",
}