import aiohttp
import asyncio
import io
import os
import json
import logging
import re
import sys
import traceback
import uuid


from contextlib import suppress 
from typing import Union

from logger import logger
from program_configs import get_config
import folder_paths

if os.name == "posix":
    try:
        print("engine.py attempting to import Aphrodite-engine...")
        from aphrodite import (
            EngineArgs,
            AphroditeEngine,
            SamplingParams,
            AsyncAphrodite,
            AsyncEngineArgs,
        )
        APHRODITE_NOT_INSTALLED = False
        print("Success!")
    except:
        print("Aphrodite-engine not installed. Only Llama CPP or API modes will run.")
        APHRODITE_NOT_INSTALLED = True

from llama_cpp import Llama, LlamaGrammar
from openai import OpenAI, AsyncOpenAI

def make_id():
    return str(uuid.uuid4())

# Define base paths and config path
base_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(base_path, "config.yml")


# Create jsons from the prompt files.
# JSON structure to prepend
json_structure = [
    {"role": "system", "content": ""},
    {"role": "user", "content": ""},
    {"role": "assistant", "content": ""},
]
prompts_dir = os.path.join(base_path, "prompts")

for filename in os.listdir(prompts_dir):
    if filename.endswith(".txt"):
        # Construct the path to the current file
        file_path = os.path.join(prompts_dir, filename)
        # Construct the new filename with .json extension
        new_filename = os.path.splitext(filename)[0] + ".json"
        new_file_path = os.path.join(prompts_dir, new_filename)

        if new_file_path not in os.listdir(prompts_dir):

            # Read the contents of the original text file
            with open(file_path, "r", encoding="utf-8") as file:
                original_content = file.read()

                # Instead of escaping, we directly assign the content
                json_structure.append({"original_content": original_content})

                # Write the JSON structure to the new file
                with open(new_file_path, "w", encoding="utf-8") as new_file:
                    json.dump(json_structure, new_file, indent=2, ensure_ascii=False)

            # Reset json_structure for the next file by removing the last element (original content)
            json_structure.pop()
        else:
            print(f"{new_file_path} already exists. Skipping.")
print("TXT to JSON conversion completed. Note: TXT files still remain.")


# Create dictionaries for the prompts and grammars from the txt files in the prompts and grammars folders. 
# The prompt names from the dictionary must match the name of the function they go to.
PROMPT_DICT = {'default_prompts': {}} 
for file_name in folder_paths.get_filename_list("prompts"):
    try:
        file_path = folder_paths.get_full_path("prompts", file_name)
        if file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                key = os.path.splitext(file_name)[0]
                PROMPT_DICT[f'{key}'] = file.read()
                #if get_config("DEBUG_MODE"):
                    #debug_dict = PROMPT_DICT[f'{key}']
                    #logger.info(f"prompt_name: {key}\nDictionary Content:\n{debug_dict}")
    except Exception as e:
        logger.exception(f"An Exception occured when creating the prompt dictionary object: {e} ")
        raise e

# Load the default prompts into the PROMPT_DICT object
for file_name in folder_paths.get_filename_list("default_prompts"):
    try:
        file_path = folder_paths.get_full_path("default_prompts", file_name)
        if file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                key = os.path.splitext(file_name)[0]
                PROMPT_DICT['default_prompts'][f'{key}'] = file.read()
    except Exception as e:
        logger.exception(f"An Exception occured when creating the default_prompts in the prompt dictionary object: {e} ")
        raise e

GRAMMAR_DICT = {}
for file_name in folder_paths.get_filename_list("grammars"):
    try:
        file_path = folder_paths.get_full_path("grammars", file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            key = os.path.splitext(file_name)[0]
            GRAMMAR_DICT[f'{key}'] = file.read()
    except Exception as e:
        logger.exception(f"An Exception occured when creating the grammar dictionary object: {e} ")
        raise e


def escape_unescaped_quotes(s):
    # Initialize a new string to store the result
    result = ""
    # Iterate through the string, keeping track of whether the current character is preceded by a backslash
    i = 0
    while i < len(s):
        # If the current character is a quote
        if s[i] == '"':
            # Check if it's the first character or if the preceding character is not a backslash
            if i == 0 or s[i - 1] != "\\":
                # Add an escaped quote to the result
                result += r"\""
            else:
                # Add the quote as is, because it's already escaped
                result += '"'
        else:
            # Add the current character to the result
            result += s[i]
        i += 1
    return result


# IMPORTANT FUNCTION: Do not change lightly as it's used by a fuck-ton of different functions.
# TODO: Test to see if function calls actually work inside this.
def format_external_text_like_f_string(external_text: str, prompt_content: str) -> str:
    # Define the regex replacement pattern. This pattern corresponds to the curly brace arguments in traditional f-strings.
    # These placeholders are expected to be in curly braces {} and can include 
    # alphanumeric characters, underscores, numeric indices in square brackets, or function calls.
    pattern = r'{([a-zA-Z0-9_]+(\[[0-9]+\]){0,2}|\w+\(\))}'
    #pattern = r'{([a-zA-Z0-9_]+(\[[0-9]+\])?|\w+\(\))}'

    def replacer(match):
        placeholder = match.group(1)

        try:
            # Replace the placeholder with the associated dictionary values from prompt_content, then return it as a string.
            value = eval(placeholder, {}, prompt_content)
            return str(value)
        except (KeyError, IndexError, TypeError, SyntaxError, NameError) as e:
            logger.exception(f"An Exception occured in format_external_text_like_f_string function using original placeholder {placeholder}: {e}")
            print("Returning the original placeholder unmodified.")
            return match.group(0)

    # Apply the replacer function to the external text. 
    # Note that external_text can be ANY text we want to treat as an f-string, not just text outside of Python. 
    # This allows the function to be applied to strings within Python as well, such as those in dictionaries and lists.
    return re.sub(pattern, replacer, external_text)

# IMPORTANT FUNCTION: Do not change lightly as it's used by everything that relies on LLM generations.
# Note: The grammar files for this need to be cleaned of comments and extraneous text before importing.
# Otherwise, the grammar will fail to load and cause the program to hang.
def load_external_prompt_and_grammar(function_name:str , grammar_name:str , prompt_content: dict) -> Union[str, str]:

    #Assume grammar is not being loaded, as it's only really used in the llama-cpp route.
    grammar = None

    # Format the function and grammar names as strings, if they're not ones already.
    grammar_name = str(grammar_name)
    function_name = str(function_name)

    print(f"Loading prompt and grammar for '{function_name}' function...")

    # Load the prompt and the grammar.
    try:
        prompt = format_external_text_like_f_string(PROMPT_DICT[f'{function_name}'], prompt_content) # Since we're importing the prompts from txt files, we can't use the regular f-string feature.
        print(f"Prompt for '{function_name}' funtion loaded successfully.")
    except Exception as e:
        logger.exception(f"An Exception occurred in '{function_name}' function while trying to import its prompt: {e}.")
        print(f"Check the prompt folder. The prompt must be a txt file named '{function_name}', and the prompt text must contain {list(prompt_content.keys())} somewhere in curly brackets.")
        logger.warning("Specified prompt failed to load. Loading default prompt...")

        # Load the default prompt if we can't load the specified one.
        try:
            prompt = format_external_text_like_f_string(PROMPT_DICT['default_prompts'][f'{function_name}'], prompt_content)
        except Exception as e:
            logger.exception(f"An Exception occurred in '{function_name}' function while trying to import its default prompt: {e}.")
            logger.error(f"'load_external_prompt_and_grammar' function failed in '{function_name}' function as it could not import its prompt and default prompt.")
            raise e
            
    try: #TODO Get rid of suppress eventually. It's apparently bad coding practice???
        with suppress(ValueError): # Since the llama-cpp-python library raises a ValueError here when loading empty grammar files, we need to surpress it.

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

async def make_async_api_call(
    prompt=None, sampling_parameters={}, url="http://127.0.0.1:8080", messages=None
):
    # Determine the endpoint based on the presence of messages
    if messages is not None:
        endpoint = "/v1/chat/completions"
        data = json.dumps(
            {
                "messages": messages,
                **sampling_parameters,  # Assuming sampling parameters can be applied to chat
            }
        )
    else:
        endpoint = "/completion"
        data = json.dumps({"prompt": prompt, **sampling_parameters})

    # Complete the URL with the chosen endpoint
    full_url = url + endpoint

    # Use aiohttp to make the async request
    async with aiohttp.ClientSession() as session:
        async with session.post(
            full_url, data=data, headers={"Content-Type": "application/json"}, ssl=False
        ) as response:
            if response.status == 200:
                # Parse the JSON response
                response_json = await response.json()
                if prompt:
                    return prompt + response_json["content"]
                else:
                    return response_json["choices"][0]["content"]
            else:
                return {"error": f"API call failed with status code: {response.status}"}

# TODO Clean up the engine wrapper comments.
class EngineWrapper:
    def __init__(self, model, 
                 api_key=get_config("API_KEY"), 
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
        
        if self.mode == "llamacpp": # Due to the way augmentoolkit was set up within Comfy, this path should never be called. But perhaps it should be???
            return await make_async_api_call(messages=messages, sampling_parameters=sampling_params)

        elif self.mode == "api":
            if get_config("DEBUG_MODE"):
                logger.info(f"\n\n\nMESSAGES\n\n\n{messages}\n")

            messages_cleaned = [
                {
                    "role": message["role"], 
                    "content": message["content"].replace("\\n","\n")
                } for message in messages
            ]

            if get_config("DEBUG_MODE"):
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


class GenerationStep:
    def __init__(self,
                 prompt_path=None, #Name of prompt txt file, relative to the Inputs directory. Now just a string! - KR
                 regex=re.compile(r'.*', re.DOTALL), # take whole completion
                 sampling_params={ # Can be overriden with override['sampling_params']
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
                 engine_wrapper=None, # This is the LLM dictionary object 'LLM'
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
            #current_dir = os.path.dirname(os.path.abspath(__file__))

            # Dynamic INPUT_DIRECTORY path
            #ideal_path = os.path.join(current_dir, '..', '..', self.prompt_folder,self.prompt_path)
            #if os.path.exists(ideal_path):
            #    full_prompt_path = ideal_path
            #else:
            #    full_prompt_path = os.path.join(current_dir, '..', '..', self.default_prompt_folder,self.prompt_path)
            if os.path.exists(folder_paths.get_prompts_directory()):
                full_prompt_path = os.path.join(folder_paths.get_prompts_directory(), self.prompt_path)
            else:
                full_prompt_path = os.path.join(folder_paths.get_default_prompts_directory(), self.prompt_path)

            if get_config("DEBUG_MODE"):
                for key in arguments.keys():
                    logger.info(f"key: \n----------------\n{key}\n----------------\n")
                for arg in arguments.items():
                    logger.info(f"arg: \n----------------\n{arg}\n----------------\n")

            try:
                if full_prompt_path.endswith(".json"): #JSON route. This is a slightly-modified version of augmentoolkit's original prompt import code.
                    with open(full_prompt_path, 'r') as pf:
                        prompt = pf.read()
                        if get_config("DEBUG_MODE"):
                            logger.info(f"Input Prompt: JSON route \n----------------\n{prompt}\n----------------\n")
                        # Code to ensure that interpolation works, but curly braces are still allowed in the input
                        # 1. Escape all curly braces
                        prompt_escaped = prompt.replace('{', '{{').replace('}', '}}')
                        # 2. Unescape curly braces that are associated with input keys
                        for key in arguments.keys():
                            prompt_escaped = prompt_escaped.replace(f"{{{{{key}}}}}", f"{{{key}}}") # Somehow this works

                        # escape the quotes in the argument values
                        for key in arguments:
                            arguments[key] = escape_unescaped_quotes(arguments[key])

                        if get_config("DEBUG_MODE"):
                            for arg in arguments.items():
                                logger.info(f"assigned arg: \n----------------\n{arg}\n----------------\n")                          
                        
                        # 3. Format
                        prompt_formatted = prompt_escaped.format(**arguments)

                        if get_config("DEBUG_MODE"):
                            logger.info(f"Formatted Prompt: JSON route \n----------------\n{prompt_formatted}\n----------------\n")

                else: # The PROMPT_DICT route, where the prompts are already loaded into ComfyUI.
                    prompt_name = os.path.splitext(os.path.basename(full_prompt_path))[0] # Extract the prompt name from its file path.

                    if get_config("DEBUG_MODE"):
                        logger.info(f"Input Prompt: Prompt Dict route \n----------------\n{prompt_name}\n----------------\n")

                    prompt_formatted, _ = load_external_prompt_and_grammar(prompt_name, "dummy_grammar", arguments) # Load the prompt with its arguments filled in.
                    
                    if get_config("DEBUG_MODE"):
                        logger.info(f"Formatted Prompt: Prompt Dict route \n----------------\n{prompt_formatted}\n----------------\n")

            except Exception as e: # Since override isn't file-based, we can use invalid path errors to signal the Generator to process self.prompt_path as a regular string.
                prompt_formatted = format_external_text_like_f_string(self.prompt_path, arguments)
                if get_config("DEBUG_MODE"):
                    logger.info(f"Formatted Prompt: Direct String route \n----------------\n{prompt_formatted}\n----------------\n")
                    logger.info(f"EXCEPTION in 'generate' function in class GenerationStep: {e}")

        except Exception as e:
            logger.exception(f"An exception occured when trying to load the prompt '{self.prompt_path}': {e}")

        times_tried = 0
        if self.completion_mode or self.engine_wrapper['type'] == "aphrodite": #Force completion mode if the LLM's type is aphrodite.
            while times_tried <= self.retries:
                try:
                    response = await self.engine_wrapper['llm'].submit_completion(prompt_formatted, self.sampling_params)
                    filtered_response = re.search(self.regex, response).group(1)
                    ret = self.output_processor(filtered_response)
                    if self.return_input_too:
                        return ret, prompt_formatted + filtered_response
                    return ret
                except Exception as e:
                    if get_config("DEBUG_MODE"):
                        logger.error(f"Error in Generation Step: {e}")
                    try:
                        if not self.engine_wrapper['type'] == "llamacpp":
                            print(f"Response:\n{response}")
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