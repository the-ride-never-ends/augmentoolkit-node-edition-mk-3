import asyncio
import os
import random
import re
#import sentiencepiece
import sys
import time
import torch
import uuid

script_dir = os.path.dirname(os.path.realpath(__file__))
custom_nodes_path = os.path.join(script_dir, "ComfyUI", "custom_nodes")
sys.path.insert(0, custom_nodes_path)

import comfy.utils
import comfy.model_management
from comfy.cli_args import args

from custom_nodes.logger import logger
#from custom_nodes import helper_functions, grammars, output_validation, logger
#from custom_nodes.logger import logger
#from custom_nodes.output_validation import extract_first_words, extract_steps, make_regenerate_answer_constrain_to_text_plan
#from custom_nodes.helper_functions import format_external_text_like_f_string

from accelerate.utils import release_memory
from datetime import datetime
from functools import partial
from llama_cpp import Llama, LlamaGrammar
from math import ceil
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Tuple

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
import folder_paths



class RunTwiceDebug:

    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "run_twice_debug": (["True","False"],),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "twice_debug"

    OUTPUT_NODE = True

    CATEGORY = "debug"

    def twice_debug(self, run_twice_debug):
        if run_twice_debug == "True":
            print("This should only print once!")
            logger.info("This should only log once as well!")
        else:
            print("This should only print once!")
            logger.info("This should only log once as well!")

        return (None,)

class LlmLoadDebug:

    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "llm_load_debug"

    OUTPUT_NODE = True

    CATEGORY = "debug"

    def llm_load_debug(self, model):
        if model is not None:
            print("The cause of this is in the 'Load Llm' node!")

        return (None,)

###############################################
#### NODE CLASS MAPPINGS AND DISPLAY NAMES ####
###############################################


NODE_CLASS_MAPPINGS = {
    "RunTwiceDebug": RunTwiceDebug,
    "LlmLoadDebug": LlmLoadDebug
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunTwiceDebug":"Run Twice Debug",
    "LlmLoadDebug": "Load LLM Debug"
}


"""
class special_instructions_prototype:
    \"\"\"
    #documentation todo
    \"\"\"
    personality_trait_schema = [
        "Axis/Non-Axis traits",
        "Big Five personality traits",
        "CAT-Personality Disorder Scale",
        "",
        "",
        "",
    ]

    schema_explanation_links = [
        "Custom trait schema by E.P. Armstrong",
        "https://en.wikipedia.org/wiki/Big_Five_personality_traits",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3400119/",
    ]

    instruction_types = [
        "special_instructions",
        "special_instructions_prototype",
        "special_instructions_prototype_2"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "personality_trait_schema": (s.personality_trait_schema,),
                "which_special_instructions_debug": (s.instruction_types,),
                "schema_explanation_link": (s.schema_explanation_links,), # Doesn't do anything but give a link explaining what each trait schema does
                #"schema_explanation_link_TEST": ({ 
                   # "ui": {
                     #   "text":
                      #      "Custom trait schema by E.P. Armstrong" if s.personality_trait_schemas == "Axis/Non-Axis traits",
                       #     "https://en.wikipedia.org/wiki/Big_Five_personality_traits" if s.personality_trait_schemas == "Big Five personality traits",
                       #     "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3400119/" if s.personality_trait_schemas == "CAT-Personality Disorder Scale",
                   # }
                #}
                #,),
                "num_of_traits": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            }
        }
        RETURN_TYPES = ("SPECIAL_INSTRUCTIONS",)
        FUNCTION = "special_instructions_prototype_2" # Switch to the regular function name.

        CATEGORY = "experimental"
        return { "ui": { "text": results } }

        def special_instructions_prototype_2(self, personality_trait_schema, num_of_traits):

            if "special_instructions":
                special_instructions()
            elif "special_instructions_prototype",:
                special_instructions_prototype()
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
    # It's not a bug, it's a feature! - KR















