from typing import Any, Optional
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler

from genai import Client, Credentials
from genai.extensions.langchain import LangChainInterface
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
# GENAI_API=<genai-api-endpoint> (optional) DEFAULT_API = "https://bam-api.res.ibm.com"
load_dotenv()


def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"


print(heading("Generate text with langchain"))


class Callback(BaseCallbackHandler):
    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"Token received: {token}")

api_key = "pak-KnqohFjgmJKou_eirLnIJMtbxdFzUYylnzScnLxVhY0"
api_endpoint = "https://bam-api.res.ibm.com"

llm = LangChainInterface(
    model_id="ibm-mistralai/mixtral-8x7b-instruct-v0-1-q",
    client=Client(credentials=Credentials(api_key, api_endpoint)),
    parameters=TextGenerationParameters(
        decoding_method=DecodingMethod.GREEDY,
        max_new_tokens=2000,
        min_new_tokens=100,
        stop_sequences=["(TOKENSTOP)"],
        return_options=TextGenerationReturnOptions(input_text=False, 
                                                    input_tokens=True)
                                                    ),
    moderations=ModerationParameters(
        # Threshold is set to very low level to flag everything (testing purposes)
        # or set to True to enable HAP with default settings
        hap=ModerationHAP(input=True, output=True, threshold=0.01)
    ),
)

prompt = """I am analyzing a time series data. A time series data is a sequence of real value observation captured over regular time interval. I am interested in detecting a temporal behavior at the end of time series. A time series can have several temporal behavior such as sudden increase, sudden drop, etc. We are only interested to detect such pattern at the end of time series.  

Can you generate a python code to detect the "Sudden Increase" temporal behavior at the end of time series. There may be multiple ways of detecting the Sudden Increase temporal behavior, you will provide three different solutions using time series analysis approach, statistical appraoch and machine learning approach. Do not generate explanation for solution.

User will provide a sensor name that is attached to a particular asset or its component. You are expected to discover the frequency, duration, or any seasonality a sensor may have while providing solution. 

We are interested in detecting sudden increase in temparature sensor time series of standby generator's engine component. 
"""
print(f"System Prompt: {prompt}")

result = llm.generate(prompts=[prompt])

print(f"Answer: {result.generations[0][0].text}")
print(result.llm_output)
print(result.generations[0][0].generation_info)