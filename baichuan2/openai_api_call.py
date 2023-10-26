# -*- coding: utf-8 -*-
# @Time : 2023/10/26 10:14
# @Author : ltm
# @Email :
# @Desc :

import openai

if __name__ == "__main__":
    openai.api_base = "http://0.0.0.0:8080/v1"
    openai.api_key = "none"

    body = {
        "messages": [
            {
              "role": "user",
              "content": "今天天气真好，我是小明"
            }
          ],
        "stream": True,
        "temperature": 0.7,
        "stop": []
    }

    for chunk in openai.ChatCompletion.create(
            model="baichuan-13b",
            **body
    ):
        if hasattr(chunk.choices[0].delta, "content"):
            print(chunk.choices[0].delta.content, end="", flush=True)
