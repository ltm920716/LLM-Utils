# -*- coding: utf-8 -*-
# @Time : 2023/10/26 11:16
# @Author : ltm
# @Email :
# @Desc : split text no more than max_step_len token by sentence

import math
import spacy
import tiktoken

mul_lang_model = "xx_sent_ud_sm"
try:
    nlp = spacy.load(mul_lang_model)
except:
    spacy.cli.download(mul_lang_model)
    nlp = spacy.load(mul_lang_model)


def get_text_token(text: str, model='gpt-3.5-turbo'):
    """
    openai model token
    Args:
        text:
        model:

    Returns:

    """
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


def other_model_token(text: str, model: str):
    pass


def get_split_paragraph_by_token(text: str, max_step_len=2000):
    """
    split text no more than max_step_len token by sentence
    Args:
        text:
        max_step_len:

    Returns:

    """
    doc = nlp(text)

    splited_text = []
    tmp_len = 0
    tmp_text = ''

    for sent in doc.sents:
        current_len = get_text_token(sent.text)
        if tmp_len + current_len < max_step_len:
            tmp_text += sent.text
            tmp_len += current_len
        else:
            if tmp_text:
                splited_text.append(tmp_text)

            if current_len > max_step_len:
                text_step = math.ceil(current_len / math.ceil(current_len / max_step_len))
                i = 0
                while text_step*i < current_len:
                    splited_text.append(sent.text[i*text_step: min((i+1)*text_step, current_len-1)])
                    i += 1

                tmp_text = ''
                tmp_len = 0
            else:
                tmp_text = sent.text
                tmp_len = current_len
    if tmp_text:
        splited_text.append(tmp_text)


