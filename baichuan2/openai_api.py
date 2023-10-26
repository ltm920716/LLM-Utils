# -*- coding: utf-8 -*-
# @Time : 2023/10/26 09:58
# @Author : ltm
# @Email :
# @Desc : integrate into openai api


import time
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from typing import List, Literal, Optional, Union
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


class EndpointHandler:
    def __init__(self, path="", device='cuda:0'):
        # load the model
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map=device, torch_dtype=dtype, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)

        generation_config = GenerationConfig.from_pretrained(path)
        generation_config.max_new_tokens = 512
        self.model.generation_config = generation_config
        logger.info(f'generate-config:\n {self.model.generation_config}')

        self.model_name = 'baichuan-13b'

    def chat(self, messages: list, temperature=0.3, max_tokens=2048, top_k=5, top_p=0.85, stream=False):
        self.model.generation_config.temperature = temperature
        if temperature == 0:
            self.model.generation_config.do_sample = False
        else:
            self.model.generation_config.do_sample = True
        self.model.generation_config.max_new_tokens = max_tokens
        self.model.generation_config.top_k = top_k
        self.model.generation_config.top_p = top_p

        response = self.model.chat(self.tokenizer, messages, stream=stream)

        return response


model_path = "local_path/Baichuan2-13B-Chat-4bits"
baichuan_model = EndpointHandler(model_path)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "trek90s"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0
    top_p: Optional[float] = 0.85
    top_k: Optional[int] = 5
    max_tokens: Optional[int] = 2048
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]]


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


llm_app = FastAPI()

llm_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@llm_app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelCard(id=baichuan_model.model_name)])


@llm_app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    return await create_chatglm_chat_completion(request)


async def create_chatglm_chat_completion(request: ChatCompletionRequest):
    global baichuan_model

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")

    messages = [message.dict() for message in request.messages]

    if request.stream:
        generate = chat_stream(messages, request)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response = baichuan_model.chat(
        messages,
        request.temperature,
        request.max_tokens,
        request.top_k,
        request.top_p,
        request.stream
    )
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def chat_stream(messages, request):
    global baichuan_model

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    response = baichuan_model.chat(
            messages,
            request.temperature,
            request.max_tokens,
            request.top_k,
            request.top_p,
            stream=True
    )

    index = 0
    for msg in response:
        new_text = msg[index:]
        # logger.info(msg)
        index = len(msg)
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(llm_app, host="0.0.0.0", port=8080)