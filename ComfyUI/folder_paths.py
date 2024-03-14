import os
import sys
import time
import logging

from logger import logger # custom logging presets.
from program_configs import get_config

# Define file extensions
# TODO: These should be configs perhaps?
supported_pt_extensions = set(['.ckpt', '.pt', '.bin', '.pth', '.safetensors'])
supported_llm_extensions = set(['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.gguf', ''])
supported_llm_grammar_extensions = set(['.json','.jsonl', '.txt',])
supported_llm_tokenizer_extensions = set(['.json','.jsonl', '.txt',])
supported_llm_prompt_extensions = set(['.json','.jsonl', '.txt',])

# Note: all file paths are relative to the path of the folder_paths.py file
base_path = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(base_path, "models")

folder_names_and_paths = {
    "checkpoints": ([os.path.join(models_dir, "checkpoints")], supported_pt_extensions),
    "configs": ([os.path.join(models_dir, "configs")], [".yaml"]),
    "loras": ([os.path.join(models_dir, "loras")], supported_pt_extensions),
    "vae": ([os.path.join(models_dir, "vae")], supported_pt_extensions),
    "clip": ([os.path.join(models_dir, "clip")], supported_pt_extensions),
    "unet": ([os.path.join(models_dir, "unet")], supported_pt_extensions),
    "clip_vision": ([os.path.join(models_dir, "clip_vision")], supported_pt_extensions),
    "style_models": ([os.path.join(models_dir, "style_models")], supported_pt_extensions),
    "embeddings": ([os.path.join(models_dir, "embeddings")], supported_pt_extensions),
    "diffusers": ([os.path.join(models_dir, "diffusers")], ["folder"]),
    "vae_approx": ([os.path.join(models_dir, "vae_approx")], supported_pt_extensions),
    "controlnet": ([os.path.join(models_dir, "controlnet"), os.path.join(models_dir, "t2i_adapter")], supported_pt_extensions),
    "gligen": ([os.path.join(models_dir, "gligen")], supported_pt_extensions),
    "upscale_models": ([os.path.join(models_dir, "upscale_models")], supported_pt_extensions),
    "custom_nodes": ([os.path.join(base_path, "custom_nodes")], []),
    "hypernetworks": ([os.path.join(models_dir, "hypernetworks")], supported_pt_extensions),
    "classifiers": ([os.path.join(models_dir, "classifiers")], {""})
}

# Output, Temp, Input, and User directories
output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
temp_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp")
input_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input")
user_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "user")

# LLM, grammar, prompts, and default prompts directories
llm_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "llm")
grammars_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "grammars")
tokenizers_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tokenizers")
prompts_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prompts")
default_prompts_directory = prompts_directory


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except Exception as e:
            logger.exception(f"ERROR: Failed to create '{directory}' directory.")
            raise e

ensure_directory_exists(output_directory)
ensure_directory_exists(temp_directory)
ensure_directory_exists(input_directory)
ensure_directory_exists(user_directory)


def set_output_directory(output_dir):
    global output_directory
    output_directory = output_dir

def set_temp_directory(temp_dir):
    global temp_directory
    temp_directory = temp_dir

def set_input_directory(input_dir):
    global input_directory
    input_directory = input_dir

def get_output_directory():
    global output_directory
    return output_directory

def get_temp_directory():
    global temp_directory
    return temp_directory

def get_input_directory():
    global input_directory
    return input_directory


# TODO Change the prompts and default prompts folder paths after API and Aphrodite have been debugged.
if get_config("PROMPTS") is not None:
    if get_config("PROMPTS") != prompts_directory:  # Use the folder_paths default if the config.yaml hasn't been changed.
        try:
            prompts_directory = get_config("PROMPTS")
            folder_names_and_paths["prompts"] = ([os.path.prompts_directory], supported_llm_prompt_extensions)
        except Exception as e:
            logger.exception("WARNING: Could not parse custom filepath for global variable 'PROMPTS' from 'config.yaml'. Defaulting to folder_paths.py presets.")


if get_config("DEFAULT_PROMPTS") is not None:
    if get_config("DEFAULT_PROMPTS") != default_prompts_directory: # Use the folder_paths default if the config.yaml hasn't been changed.
        try:
            default_prompts_directory = get_config("DEFAULT_PROMPTS")
            folder_names_and_paths["default_prompts"] = ([os.path.default_prompts_directory], supported_llm_prompt_extensions)
        except Exception as e:
            logger.exception("WARNING: Could not parse custom filepath for global variable 'DEFAULT_PROMPTS' from 'config.yaml'. Defaulting to folder_paths.py presets.")


if get_config("INPUT") is not None:
    if get_config("INPUT") != input_directory: # Use the folder_paths default if the config.yaml hasn't been changed.
        try:
            set_input_directory(get_config("INPUT"))
        except Exception as e:
            logger.exception("WARNING: Could not parse custom filepath for global variable 'INPUT' from 'config.yaml'. Defaulting to folder_paths.py presets.")


if get_config("OUTPUT") is not None:
    if get_config("OUTPUT") != output_directory: # Use the folder_paths default if the config.yaml hasn't been changed.
        try:
            set_output_directory(get_config("OUTPUT"))
        except Exception as e:
            logger.exception("WARNING: Could not parse custom filepath for global variable'OUTPUT' from 'config.yaml'. Defaulting to folder_paths.py presets.")


def get_default_prompts_directory():
    global default_prompts_directory
    return default_prompts_directory

def get_prompts_directory():
    global prompts_directory
    return prompts_directory

def get_llm_directory():
    global llm_directory
    return llm_directory

def get_grammars_directory():
    global grammars_directory
    return grammars_directory

def get_tokenizers_directory():
    global tokenizers_directory
    return tokenizers_directory

def get_prompts_directory():
    global prompts_directory
    return prompts_directory

def set_raw_text_name(name):
    global raw_text_name
    raw_text_name = name

def get_raw_text_name():
    global raw_text_name
    return raw_text_name

def set_loaded_llm_name(name):
    global loaded_llm_name
    loaded_llm_name = name

def get_loaded_llm_name():
    global loaded_llm_name
    return loaded_llm_name


filename_list_cache = {}

###################################

#NOTE: used in http server so don't put folders that should not be accessed remotely
def get_directory_by_type(type_name):
    if type_name == "output":
        return get_output_directory()
    if type_name == "temp":
        return get_temp_directory()
    if type_name == "input":
        return get_input_directory()
    return None


# determine base_dir rely on annotation if name is 'filename.ext [annotation]' format
# otherwise use default_path as base_dir
def annotated_filepath(name):
    if name.endswith("[output]"):
        base_dir = get_output_directory()
        name = name[:-9]
    elif name.endswith("[input]"):
        base_dir = get_input_directory()
        name = name[:-8]
    elif name.endswith("[temp]"):
        base_dir = get_temp_directory()
        name = name[:-7]
    else:
        return name, None

    return name, base_dir


def get_annotated_filepath(name, default_dir=None):
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        if default_dir is not None:
            base_dir = default_dir
        else:
            base_dir = get_input_directory()  # fallback path

    return os.path.join(base_dir, name)


def exists_annotated_filepath(name):
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        base_dir = get_input_directory()  # fallback path

    filepath = os.path.join(base_dir, name)
    return os.path.exists(filepath)


def add_model_folder_path(folder_name, full_folder_path):
    global folder_names_and_paths
    if folder_name in folder_names_and_paths:
        folder_names_and_paths[folder_name][0].append(full_folder_path)
    else:
        folder_names_and_paths[folder_name] = ([full_folder_path], set())

def get_folder_paths(folder_name):
    return folder_names_and_paths[folder_name][0][:]

def recursive_search(directory, excluded_dir_names=None):
    if not os.path.isdir(directory):
        return [], {}

    if excluded_dir_names is None:
        excluded_dir_names = []

    result = []
    dirs = {directory: os.path.getmtime(directory)}
    for dirpath, subdirs, filenames in os.walk(directory, followlinks=True, topdown=True):
        subdirs[:] = [d for d in subdirs if d not in excluded_dir_names]
        for file_name in filenames:
            relative_path = os.path.relpath(os.path.join(dirpath, file_name), directory)
            result.append(relative_path)
        for d in subdirs:
            path = os.path.join(dirpath, d)
            dirs[path] = os.path.getmtime(path)
    return result, dirs

def filter_files_extensions(files, extensions):
    return sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in extensions or len(extensions) == 0, files)))


def get_full_path(folder_name, filename):
    global folder_names_and_paths
    if folder_name not in folder_names_and_paths:
        return None
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path

    return None

def get_filename_list_(folder_name):
    global folder_names_and_paths
    output_list = set()
    folders = folder_names_and_paths[folder_name]
    output_folders = {}
    for x in folders[0]:
        files, folders_all = recursive_search(x, excluded_dir_names=[".git"])
        output_list.update(filter_files_extensions(files, folders[1]))
        output_folders = {**output_folders, **folders_all}

    return (sorted(list(output_list)), output_folders, time.perf_counter())

def cached_filename_list_(folder_name):
    global filename_list_cache
    global folder_names_and_paths
    if folder_name not in filename_list_cache:
        return None
    out = filename_list_cache[folder_name]

    for x in out[1]:
        time_modified = out[1][x]
        folder = x
        if os.path.getmtime(folder) != time_modified:
            return None

    folders = folder_names_and_paths[folder_name]
    for x in folders[0]:
        if os.path.isdir(x):
            if x not in out[1]:
                return None

    return out

def get_filename_list(folder_name):
    out = cached_filename_list_(folder_name)
    if out is None:
        out = get_filename_list_(folder_name)
        global filename_list_cache
        filename_list_cache[folder_name] = out
    return list(out[0])

def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
    def map_filename(filename):
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[:prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1:].split('_')[0])
        except:
            digits = 0
        return (digits, prefix)

    def compute_vars(input, image_width, image_height):
        input = input.replace("%width%", str(image_width))
        input = input.replace("%height%", str(image_height))
        return input

    filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = os.path.join(output_dir, subfolder)

    if os.path.commonpath((output_dir, os.path.abspath(full_output_folder))) != output_dir:
        err = "**** ERROR: Saving image outside the output folder is not allowed." + \
              "\n full_output_folder: " + os.path.abspath(full_output_folder) + \
              "\n         output_dir: " + output_dir + \
              "\n         commonpath: " + os.path.commonpath((output_dir, os.path.abspath(full_output_folder))) 
        print(err)
        raise Exception(err)

    try:
        counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix
