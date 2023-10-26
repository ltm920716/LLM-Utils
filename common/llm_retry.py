# -*- coding: utf-8 -*-
# @Time : 2023/10/26 11:15
# @Author : ltm
# @Email :
# @Desc : openai llm demo

import openai
import logging
from loguru import logger
from typing import Any, Callable
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

max_retries = 5


def _create_retry_decorator() -> Callable[[Any], Any]:
    min_seconds = 1
    max_seconds = 120
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
                | retry_if_exception_type(openai.error.OpenAIError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(**kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator()

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        return openai.ChatCompletion.create(**kwargs)

    return _completion_with_retry(**kwargs)