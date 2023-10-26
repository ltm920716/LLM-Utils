# -*- coding: utf-8 -*-
# @Time : 2023/10/26 09:57
# @Author : ltm
# @Email :
# @Desc : test in V100 & A100 for streaming

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from loguru import logger

app = FastAPI()

model_path = "local_path/Baichuan2-13B-Chat-4bits"
device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16, trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained(model_path)
generation_config.max_new_tokens = 400
model.generation_config = generation_config
logger.info(f'generate-config:\n {model.generation_config}')


class TextGenerationInput(BaseModel):
    messages: list
    temperature: float
    stream: bool


class TextGenerationOutput(BaseModel):
    generated_text: str


@app.post("/chat", response_model=TextGenerationOutput)
async def generate_text(input_data: TextGenerationInput):
    logger.info(f'Input: {input_data.messages}')
    assert input_data.temperature >= 0
    model.generation_config.temperature = input_data.temperature
    if input_data.temperature == 0:
        model.generation_config.do_sample = False
    else:
        model.generation_config.do_sample = True

    response = model.chat(tokenizer, input_data.messages, stream=input_data.stream)
    if input_data.stream:
        def msg_yield():
            index = 0
            msg = ''
            for msg in response:
                yield msg[index:]
                index = len(msg)

            logger.info(f'Output: {msg}')

        return EventSourceResponse(msg_yield())
    else:
        logger.info(f'Output: {response}')
        # torch.mps.empty_cache()
        return TextGenerationOutput(generated_text=response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)