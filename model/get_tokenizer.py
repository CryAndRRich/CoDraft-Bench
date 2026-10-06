from transformers import AutoTokenizer, PreTrainedTokenizer

from config.config_data import CONFIG_DATA


def get_tokenizer(model_name: str, add_class_tokens: bool = False) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if add_class_tokens:
        tokenizer.add_tokens(CONFIG_DATA.CLASS_TOKENS, special_tokens=True)
    return tokenizer
