import os
from typing import Any, Optional
from uuid import UUID

from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler

from genai.credentials import Credentials
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.schemas.generate_params import HAPOptions, ModerationsOptions

LLMsets = ['ibm/granite-13b-instruct-v2',
        'meta-llama/llama-2-70b',
        'google/flan-ul2',
        'thebloke/mixtral-8x7b-instruct-v0-1-gptq']

api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_endpoint = "https://bam-api.res.ibm.com"

for llmmodel in LLMsets:
    llm = LangChainInterface(
        model=llmmodel,
        credentials=Credentials(api_key, api_endpoint),
        params=GenerateParams(
            decoding_method="sample",
            max_new_tokens=10,
            min_new_tokens=1,
            stream=True,
            temperature=0.5,
            top_k=50,
            top_p=1,
            moderations=ModerationsOptions(
                # Threshold is set to very low level to flag everything (testing purposes)
                # or set to True to enable HAP with default settings
                hap=HAPOptions(input=True, output=True, threshold=0.01)
            ),
        ),
    )

    result = llm.generate(
        prompts=["Tell me about IBM."],
    )
    print(f"LLM: {llmmodel}, Response: {result.generations[0][0].text}")
