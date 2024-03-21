
from logger import logger
from program_configs import get_config
from engine import EngineWrapper


# Note: This file is fairly sparse, as most of the backend for the API calls happens within the EngineWrapper from engine.py
# TODO Test with other API services and front-ends (e.g. Ooba, Kobaldcpp)

try:
    print("augmentoolkit_api_functions.py attempting to import OpenAI api...")
    import openai
    OPENAI_NOT_INSTALLED = False  
    print("Success!")
except:
    print("Note: OpenAI client not installed.")
    OPENAI_NOT_INSTALLED = True

try:
    print("augmentoolkit_api_functions.py attempting to import TogetherAI api...")
    import together
    TOGETHER_NOT_INSTALLED = False
    print("Success!")
except:
    logger.info("Note: Together.ai client not installed.")
    TOGETHER_NOT_INSTALLED = True

class ChatGPT:
    """
    Load a model from OpenAI via its API.
    Modified from: https://github.com/xXAdonesXx/NodeGPT/blob/main/API_Nodes/ChatGPT.py
    :param model_name:
    :param model_name:
    :return LLM:
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "gpt-4"})
            },
            "optional": {
                "open_ai_api_key": ("STRING", {"default": None}),
                #"open_ai_api_base_url": ("STRING", {"default": None}),
            }
        }

    RETURN_TYPES = ("LLM",)
    FUNCTION = "execute"
    CATEGORY = "augmentoolkit_functions/loaders"

    def execute(self, model_name, open_ai_api_key):
        # Load the api key from the 'config.yml' file if a custom key is not present
        if open_ai_api_key is None:
            logger.info("'open_ai_api_key' argument not specified. Defaulting to API key from config.yaml.")
            try:
                engine_wrapper = EngineWrapper(model=model_name, mode="api", api_key=get_config("API_KEY"),)
                config_list = [
                    {
                        'llm': engine_wrapper,
                        'type': 'api',
                        'api_subtype': 'openai'
                    }
                ]
            except Exception as e:
                logger.exception(f"An Exception occured when trying to import the OpenAI API key: {e}")
                print("This may have occured because the API_KEY or BASE_URL in the 'config.yaml' file does not exist or is for another service.")
                raise e
        else:
            try:
                engine_wrapper = EngineWrapper(model=model_name, mode="api", api_key=open_ai_api_key,)
                config_list = [
                    {
                        'llm': engine_wrapper,
                        'type': 'api',
                        'api_sub_type': 'openai',
                        'prompt': None,
                        'sampling_params': None
                    }
                ]
            except Exception as e:
                logger.exception(f"An Exception occured in 'execute' function in class 'ChatGPT' while trying to import the specified OpenAI API key: {e}")
                raise e

        return ({"LLM": config_list},)


class LM_Studio:
    """
    Load a model from LM Studio via its API.
    Modified from: https://github.com/xXAdonesXx/NodeGPT/blob/main/API_Nodes/ChatGPT.py
    :param model_name:
    :param model_name:
    :return LLM:
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": None}),
                "api_key": ("STRING", {"default": "NULL"}),
                "api_base": ("STRING", {"default": "http://localhost:1234/v1"}),
            }
        }

    RETURN_TYPES = ("LLM",)
    FUNCTION = "execute"
    CATEGORY = "augmentoolkit_functions/loaders"

    def execute(self, model_name, api_key, api_base):
        try:
            engine_wrapper = EngineWrapper(model=model_name, mode="api", api_key=api_key, base_url=api_base,)
            config_list = [
                {
                    'llm': engine_wrapper,
                    'type': 'api',
                    'api_subtype': 'lm_studio',
                    'api_key': api_key,
                    'base_url': api_base,
                    'prompt': None,
                    'sampling_params': None,
                }
            ]
        except Exception as e:
            logger.exception(f"An Exception occured in 'execute' function in class 'LM_Studio' : {e}")
            raise e

        return ({"LLM": config_list},)


# TODO figure out what api_type means.
# TODO Write function documentation.
class Ollama:
    """
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "ollama/mistral"}),
                "api_type": ("STRING", {"default": "litellm"}),
                "api_base": ("STRING", {"default": "http://0.0.0.0:8000"})
            }
        }

    RETURN_TYPES = ("LLM",)
    FUNCTION = "execute"
    CATEGORY = "augmentoolkit_functions/loaders"

    def execute(self, model_name, api_type, api_base):
        try:
            engine_wrapper = EngineWrapper(model=model_name, mode="api", base_url=api_base,)
            config_list = [
                {
                    'model': engine_wrapper,
                    'type': 'api', 
                    'apit_subtype': 'ollama',
                    'api_type': api_type, #Holdover from the original code. Kept for backwards compatability, if it's even feasible.
                    'api_base': api_base,
                    'prompt': None,
                    'sampling_params': None,
                }
            ]
        except Exception as e:
            logger.exception(f"An Exception occured in 'execute' function in class 'Ollama' : {e}")
            raise e

        return ({"LLM": config_list},)

"""
class Mistral:
    pass


class KobaldCpp:
    pass
"""

NODE_CLASS_MAPPINGS = {
    "ChatGPT": ChatGPT,
    #"KobaldCpp": KobaldCpp,
    #"Mistral": Mistral,
    "LM_Studio": LM_Studio,
    "Ollama": Ollama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatGPT": "Load Model (OpenAI)",
    #"KobaldCpp": "Load Model (Kobald CPP)",
    #"Mistral": "Load Model (Mistral)",
    "LM_Studio": "Load Model (LM Studio)",
    "Ollama": "Load Model (Ollama)",
}

