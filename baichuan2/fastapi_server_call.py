# -*- coding: utf-8 -*-
# @Time : 2023/10/26 10:14
# @Author : ltm
# @Email :
# @Desc :

import requests
import sseclient


url = "http://0.0.0.0:8081/chat"

messages = [
    {"role": "user", "content": "你好，我是小明"}
]

data = {
    "messages": messages,
    'temperature': 0.3,
    'stream': True
}

for stream_resp in sseclient.SSEClient(
    requests.request(
        method='post', url=url, json=data, stream=True
    )
).events():
    tool_resp = stream_resp.data
    print(tool_resp, end='', flush=True)