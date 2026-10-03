from transformers import AutoTokenizer

from config.config_data import CONFIG_DATA

def get_tokenizer(model_name, add_class_tokens=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if add_class_tokens:
        # special_tokens=True keeps each marker whole instead of letting the
        # SentencePiece model split it.
        tokenizer.add_tokens(CONFIG_DATA.CLASS_TOKENS, special_tokens=True)
    return tokenizer
