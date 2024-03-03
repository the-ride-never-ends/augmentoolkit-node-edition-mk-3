from transformers import AutoTokenizer
import re
import time
from tqdm import tqdm
import logging
from datetime import datetime
import os
import nltk
import random
import asyncio
import traceback

from custom_nodes.logger import logger

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

import folder_paths

######################
#### NODE CLASSES ####
######################

class ChunkSentence:
    """
    This function takes a plaintext (txt) file and chunks it into sentences.
    :param raw_text: text from a plaintext (txt) file, loaded in from the InputTextLoaderSimple node.
    :param tokenizer: sentiencepiece tokenizer (original: SentencePiece).
    :param max_token_length: The maximum token length for a chunk of sentences.
    :return cleaned_paragraphs: List of tuples of sentence chunks with source text information.
    """
    def __init__(self):
        self.source_name = folder_paths.get_raw_text_name()

    @staticmethod
    def sentence_chunking_algorithm(source_name, sentences, tokenizer, max_token_length):
        try:
            sentence_chunks_with_source = []
            current_chunk = []
            token_count = 0

            for sentence in tqdm(sentences, desc=f"Processing {source_name}"):

                sentence_token_count = len(tokenizer.encode(sentence))
                #logger.info(f"sentence_token_count: {sentence_token_count}")

                if token_count + sentence_token_count <= max_token_length:
                    current_chunk.append(sentence)
                    token_count += sentence_token_count
                else:
                    sentence_chunks_with_source.append((" ".join(current_chunk), source_name.replace(".txt", "")))
                    current_chunk = [sentence]
                    token_count = sentence_token_count

            # Add the last chunk if it exists
            if current_chunk:
                sentence_chunks_with_source.append((" ".join(current_chunk), source_name.replace(".txt", "")))

            return sentence_chunks_with_source

        except Exception as e:
            logger.exception(f"An Exception occured in sentence_chunking_algorithm function in class ChunkSentence: {e}")

    @staticmethod
    def replace_unicode_escapes(text):
        # This function will replace all occurrences of Unicode escape sequences
        # in the form of \uXXXX (where X is a hexadecimal digit) with the corresponding Unicode character.
        return re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)

    @staticmethod
    def fix_text(to_replace_arr, text_to_fix):
        try:
            fixed_text = text_to_fix
            for replace_pair in to_replace_arr:
                # Fix the text based
                fixed_text = fixed_text.replace(replace_pair[0], replace_pair[1])

                # Replace Unicode escape sentences with their corresponding Unicode characters.
                fixed_text = ChunkSentence.replace_unicode_escapes(fixed_text)

                #logger.info(f"Fixed text: {fixed_text}Fixed text")
                #time.sleep(5)

            return fixed_text
        except Exception as e:
            logger.exception(f"An Exception occurred in fix_text function in class ChunkSentence: {e}")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_text": ("RAW_TEXT",),
                "tokenizer": ("TOKENIZER",),
                "max_token_length": ("INT", {"default": 400, "min": 400, "max": 10000, "step": 1}),
            }
        }
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("paragraph_tuples",)
    FUNCTION = "return_clean_paragraphs"

    CATEGORY = "input_preparation"

    def return_clean_paragraphs(self, raw_text, tokenizer, max_token_length):
        # Initialize the sentence_chunks list.
        sentence_chunks = []

        # Remove Gutenberg header and footer
        raw_text = re.sub(r"^.*?START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*$\n", "", raw_text, flags=re.MULTILINE,)
        raw_text = re.sub(r"^.*?END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*$\n", "", raw_text, flags=re.MULTILINE,)

        #print(f"{raw_text}: RAW TEXT VARIABLE DEBUG")
        #time.sleep(10)

        #Sentence tokenize the text (SentencePiece)
        tokenized_text = sent_tokenize(raw_text)

        #logger.info(f"{tokenized_text}: TOKENIZED_TEXT DEBUG, TYPE: {type(tokenized_text)}")
        #time.sleep(10)

        # Chunk the paragraphs and put them into the list.
        sentence_chunks += self.sentence_chunking_algorithm(self.source_name, tokenized_text, tokenizer, max_token_length)

        # DEBUGGING: Create a string that contains sampled elements from each chunk raw.
        #logger.info(f"SENTENCE_CHUNKS DEBUG, TYPE: {type(tokenized_text)}: {sentence_chunks}: SENTENCE_CHUNKS DEBUG, TYPE: {type(tokenized_text)}")
        #time.sleep(5)

        #Initialize the conversions to clean up the text, namely remove the newline symbol, closing gaps, and removing UTF-8 import errors.
        #TODO: Maybe turn this into an external dictionary if it gets super long???
        conversions = [
            ("\n", " "), ("  ", " "), ("\ufeff","")
        ]

        #Clean up the text in the sentences chunks.
        cleaned_paragraphs = [
            (self.fix_text(conversions, seq[0]), seq[1]) for seq in sentence_chunks
        ]

        logger.info("Paragraphs successfully cleaned.")
        time.sleep(3)
        #logger.info(f"cleaned_paragraphs: {cleaned_paragraphs}: cleaned_paragraphs")
        #time.sleep(10)

        #logger.info(f"CLEANED_PARAGRAPHS DEBUG, TYPE: {type(tokenized_text)}: {sentence_chunks}: CLEANED_PARAGRAPHS DEBUG, TYPE: {type(tokenized_text)}")
        #time.sleep(5)

        return (cleaned_paragraphs,)


NODE_CLASS_MAPPINGS = {
    #Input Preparation
    "ChunkSentence":ChunkSentence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    #Input Preparation
    "ChunkSentence":"Chunk Plaintext File into Paragraphs",
}