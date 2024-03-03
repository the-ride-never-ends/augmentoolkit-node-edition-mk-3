import asyncio
import importlib
import logging
import os
import time
import torch
import traceback

from transformers import AutoTokenizer
from datetime import datetime
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
#import sentiencepiece


import comfy.utils
import comfy.model_management
import folder_paths
from comfy.cli_args import args
from custom_nodes import helper_functions
from custom_nodes.logger import logger


"""
def load_grammar(self, grammar_name):
    grammar_path = folder_paths.get_full_path("grammars", grammar_name)
    try:
        opened_grammar = open(f"{grammar_path}",r)
        grammar = LlamaGrammar.from_string(opened_grammar)
    except Exception as e:
        print(f"An Exception occured for load_grammar function: {e}")
    return (grammar,)
"""

######################
#### NODE CLASSES ####
######################

class TokenizerLoaderSimple: # TODO Figure out how the fuck to actually load the tokenizer.


    sentencepiece_tokenizer = [
        "Gryphe/MythoMax-L2-13b"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tokenizer_name": (folder_paths.get_folder_paths("tokenizers"),),
            }
        }

    RETURN_TYPES = ("TOKENIZER",)
    FUNCTION = "load_tokenizer"

    CATEGORY = "llm_loaders"

    def load_tokenizer(self, tokenizer_name):
        #tokenizer_path = folder_paths.get_full_path("tokenizers", tokenizer_name)
        dummy_variable = tokenizer_name
        batch_file_name = os.getenv('BATCH_FILE_NAME')
        try:
            #os.path.join(folder_paths.get_input_directory(),text_name)
            if batch_file_name:
                tokenizer = AutoTokenizer.from_pretrained("Gryphe/MythoMax-L2-13b")
            else:
                tokenizer = AutoTokenizer.from_pretrained(os.path.join(folder_paths.get_tokenizers_directory(), "llama-tokenizer\\"))
            logger.info(f"Sentencepiece Tokenizer succesfully loaded.")

        except Exception as e:
            logger.exception(f"An Exception occured for load_tokenizer function: {e}")

        return (tokenizer,)

class LlmLoaderSuperSimple: 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("llm"),),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "llm_loaders"

    def load_model(self, model_name):
        try:
            model = Llama(
                model_path=folder_paths.get_full_path("llm", model_name),
                offload_kqv=True,
                n_ctx=12000,
                rope_freq_scale=0.33,
                n_gpu_layers=100,
                verbose=False,
            )

            folder_paths.set_loaded_llm_name(model_name)
            logger.info(f"global variable loaded_llm_name set to: {folder_paths.get_loaded_llm_name()}")
            time.sleep(3)

            logger.info(f"Model {model_name} succesfully loaded.")
            time.sleep(3)

        except Exception as e:
            logger.exception(f"An Exception occured in load_model function in class LlmLoaderSimple: {e}")

        return (model,)

# TODO: Rename all the INITIALIZED_LLM categories to just LLM so that it can be plug-in-play with ComfyUI-Llama
class LLM_Load_Model:
    """
    Load a llama.cpp model from model_path.
    (easy version)
    From ComfyUI-Llama: https://github.com/daniel-lewis-ab/ComfyUI-Llama/blob/main/Llama.py
    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("llm"), ), 
            },
            "optional": {
                "n_ctx": ("INT", {"default": 0, "step":512, "min":0}),
            }
        }

    # ComfyUI will effectively return the Llama class instantiation provided by execute() and call it an LLM
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "debug"

    def execute(self, model_name:str, n_ctx:int):

        # basically just calls __init__ on the Llama class
        model_path = folder_paths.get_full_path("llm", model_name)

        try:
            model = Llama(
                model_path=model_path,
                n_gpu_layers=100,
                n_ctx=12000, 
                seed=-1,
            )

        except ValueError:
            logger.exception("The model path does not exist. Perhaps hit Ctrl+F5 and try reloading it.")

        return (model,)

# TODO: Get speculative sampling to actually work. Right now, the n-gram version outputs unreadable garbage.
class LlmLoaderSimple: # TODO Setup the config so that it can pull the LLMs from outside the program i.e. if they're in a folder in Oobabooga or something.
    """



    """



 # Class variable to hold the singleton model instance
    # This is necessary to prevent someone from loading in the same model twice, eating up all their RAM and VRAM in the process.
    _singleton_model_instance = None

    def __init__(self):
        # Instance variable to hold the model name
        self.model_name = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("llm"),),
                "n_gqa_arg": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1},), #llamacpppython removed this, so this argument is unnecessary... but maybe not?
                "offload_kqv_arg": (["True", "False"],),
                "n_ctx_arg": ("INT", {"default": 12000, "min": 256, "max": 1000000, "step": 1},), # ONE MILLION CONTEXT!!!
                "rope_freq_scale_arg": ("FLOAT", {"default": 0.33, "min": 0.00, "max": 1.00, "step": 0.01},),
                "n_gpu_layers_arg": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1},),
                "verbose_arg": (["False", "True"],),
                "speculative_drafting_arg": (["True", "False"],),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "llm_loaders"

    def load_model(self, model_name, n_gqa_arg, offload_kqv_arg, n_ctx_arg, rope_freq_scale_arg, n_gpu_layers_arg, verbose_arg, speculative_drafting_arg):
        # Update instance's model name
        self.model_name = model_name

        # Check if we already have a model loaded.
        if LlmLoaderSimple._singleton_model_instance is not None:
            logger.info(f"Model {model_name} already loaded, using existing instance.")
            return LlmLoaderSimple._singleton_model_instance

        # Load the model with the specified arguments.
        if LlmLoaderSimple._singleton_model_instance is None:
            try:
                model = Llama(
                    model_path=folder_paths.get_full_path("llm", model_name),
                    n_gqa=n_gqa_arg,
                    n_gpu_layers=n_gpu_layers_arg,
                    offload_kqv=True if offload_kqv_arg == "True" else False,
                    n_ctx=n_ctx_arg,
                    rope_freq_scale=rope_freq_scale_arg,
                    verbose=True if verbose_arg == "True" else False,
                    draft_model=None if speculative_drafting_arg == "True" else None, # LlamaPromptLookupDecoding(num_pred_tokens=10)
                )

                # Update the class variable with the new model instance
                LlmLoaderSimple._singleton_model_instance = model

                # Set the global variable for the model's name.
                folder_paths.set_loaded_llm_name(model_name)
                logger.info(f"global variable loaded_llm_name set to: {folder_paths.get_loaded_llm_name()}")
                time.sleep(1)

                logger.info(f"Model {model_name} succesfully loaded.")
                time.sleep(1)

            except Exception as e:
                logger.exception(f"An Exception occured in load_model function in class LlmLoaderSimple: {e}")
                LlmLoaderSimple._singleton_model_instance = None  # Ensure no partial instance is kept
                raise e

        else:
            logger.error(f"ERROR: Specified model {model_name} already loaded.")

        return (model,)

class InputTextLoaderSimple:
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

    CATEGORY = "llm_loaders"
    
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

"""
class LlmLoaderAphroditeSimple:

    _singleton_model_instance = None

    def __init__(self):
        # Instance variable to hold the model name
        self.model_name = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("llm"),),
                "n_gqa_arg": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1},),
                "offload_kqv_arg": (["True", "False"],),
                "n_ctx_arg": ("INT", {"default": 12000, "min": 256, "max": 1000000, "step": 1},), # ONE MILLION CONTEXT!!!
                "rope_freq_scale_arg": ("FLOAT", {"default": 0.33, "min": 0.00, "max": 1.00, "step": 0.01},), # Haha, I have no idea what I'm doing.
                "n_gpu_layers_arg": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1},),
                "verbose_arg": (["False", "True"],),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "debug"

class LlmLoaderApiSimple:
    _singleton_model_instance = None

    def __init__(self):
        # Instance variable to hold the model name
        self.model_name = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("llm"),),
                "n_gqa_arg": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1},),
                "offload_kqv_arg": (["True", "False"],),
                "n_ctx_arg": ("INT", {"default": 12000, "min": 256, "max": 1000000, "step": 1},), # ONE MILLION CONTEXT!!!
                "rope_freq_scale_arg": ("FLOAT", {"default": 0.33, "min": 0.00, "max": 1.00, "step": 0.01},), # Haha, I have no idea what I'm doing.
                "n_gpu_layers_arg": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1},),
                "verbose_arg": (["False", "True"],),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "debug"
"""


NODE_CLASS_MAPPINGS = {
    #Loaders
    #"GrammarLoaderSimple": GrammarLoaderSimple,
    "InputTextLoaderSimple": InputTextLoaderSimple,
    "LlmLoaderSimple": LlmLoaderSimple,
    "LlmLoaderSuperSimple": LlmLoaderSuperSimple,
    "LLM_Load_Model": LLM_Load_Model,
    #"PromptTextLoaderSimple": PromptTextLoaderSimple,
    #"LlmLoaderApiSimple": LlmLoaderApiSimple,
    #"LlmLoaderAphroditeSimple": LlmLoaderAphroditeSimple,
    "TokenizerLoaderSimple": TokenizerLoaderSimple,


    #Input Preparation
    #"ChunkSentence":ChunkSentence,
    #"TextEncode": TextEncode, 

    #Output Generation
    #"GenerateNewQuestion": GenerateNewQuestion,
    #"CreateCharacterCardPlanManyTuples": CreateCharacterCardPlanManyTuples,
    #"CreateScenarioPlanManyTuples": CreateScenarioPlanManyTuples,
    #"MakeMultiturnCharacter": MakeMultiturnCharacter,
    #"MakeMuliturnScenario": MakeMuliturnScenario,
    #"MakeMultirunConversationInfo": MakeMultirunConversationInfo,
    #"MakeMultiturnConversation": MakeMultiturnConversation,

    #Output Validation
    #"EnsureMultipleAnswersAreSame": EnsureMultipleAnswersAreSame,
    #"JudgeParagraph": JudgeParagraph,
    #"FilterAndGraph": FilterAndGraph,
    #"CheckAnswer": CheckAnswer,
    #"CheckAnsewerRelevancyWithText": CheckAnsewerRelevancyWithText,
    #"CheckQuestion": CheckQuestion,
    #"CheckQATupleContext": CheckQATupleContext,
    #"IdentifyDuplicates": IdentifyDuplicates,
    #"ArmchairPsychologist": ArmchairPsychologist,

    #Output Correction
    #"FixText":FixText,
    #"FormatQATuples": FormatQATuples,
    #"FormatQATuplesNoQuotes": FormatQATuplesNoQuotes,

    #Helper Functions
    #"GroupByText": GroupByText,
    #"ExtractName": ExtractName,
    #"ExtractConversation": ExtractConversation,
    #read_json_files_info
    # Examples
    #"CheckpointLoader": CheckpointLoader,
    #"DiffusersLoader": DiffusersLoader,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    #LLM Loaders
    #"GrammarLoaderSimple": "Load Grammar",
    "InputTextLoaderSimple": "Load Input Text (Simple)",
    "LlmLoaderSimple": "Load LLM (llama-cpp-python)",
    "LlmLoaderSuperSimple": "Load LLM (llama-cpp-python) (Simple)",
    "LLM_Load_Model": "Load LLM (ComfyUI-Llama) (Debug)",
    #"LlmLoaderApiSimple": "Load LLM (API)",
    #"LlmLoaderAphroditeSimple": "Load LLM (Aphrodite-engine)",
    #"PromptTextLoaderSimple": "Load Prompt",
    "TokenizerLoaderSimple": "Load Tokenizer",


    #Input Preparation
    #"ChunkSentence":"Chunk Sentences",
    #"TextEncode": "Encode Text", 
    #Output Generation
    #"GenerateNewQuestion": "Generate New Question",
    #"CreateCharacterCardPlanManyTuples": "Create Character Card Plan (Many Tuples)",
    #"CreateScenarioPlanManyTuples": "Create Scenario Plan (Many Tuples)",
    #"MakeMultiturnCharacter": "Make Multi-turn Character",
    #"MakeMuliturnScenario": "Make Multi-turn Scenario",
    #"MakeMultirunConversationInfo": "Make Multi-turn Conversation Info",
    #"MakeMultiturnConversation": "Make Multi-turn Conversation",
    #Output Validation
    #"EnsureMultipleAnswersAreSame": "Ensure Multiple Answers are the Same",
    #"JudgeParagraph": "Judge Paragraph",
    #"FilterAndGraph": "Filter and Graph (Filter and Graph what exactly?)",
    #"CheckAnswer": "Check Answer",
    #"CheckAnsewerRelevancyWithText": "Check Answer Relevancy with Text",
    #"CheckQuestion": "Check Question",
    #"CheckQATupleContext": "Check QA Tuple Context",
    #"IdentifyDuplicates": "Identify Duplicates",
    #"ArmchairPsychologist": "Check Character Card Personality Adherence",
    #Output Correction
    #"FixText":"Fix Text",
    #"FormatQATuples": "Format QA Tuples",
    #"FormatQATuplesNoQuotes": "Format QA Tuples (No Quotes)",
    #Helper Functions
    #"GroupByText": "Group by Text",
    #"ExtractName": "Extract Name",
    #"ExtractConversation": "Extract Conversation",
}
















