import os

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from genai.credentials import Credentials
from genai.extensions.langchain import LangChainInterface
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.schemas import ChatOptions, GenerateParams, ReturnOptions
from genai.schemas.generate_params import HAPOptions, ModerationsOptions

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_endpoint = "https://bam-api.res.ibm.com"

SystemPrompt = """
You are a helpful, respectful, and honest assistant. You will be provided a one line description that 
include industrial asset name and may include some configuration such as component name or configuration. You need  
to identify the device name which represent an industrial asset from the given description and then 
generate a device type and short device description. Provide answer in python json string with following keys : 
asset_class, asset_category, device_name, device_type and device_description. 

If you don't know the answer to a question, please don't share false information. Your answers should not 
include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that 
your responses are socially unbiased and positive in nature.
"""

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v01-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
]

# Is this question for Subject Matter Expert?
# Is this question for Data Scientist?

import pandas as pd

df = pd.read_csv("./AssetClassInfo.csv")
instructions = list(df["TypeData.GenCompType"].unique())

llm = LangChainInterface(
    model=LLMsets[1],
    credentials=Credentials(api_key, api_endpoint),
    params=GenerateParams(
        decoding_method="greedy",
        max_new_tokens=500,
        min_new_tokens=200,
        temperature=0.5,
        top_k=50,
        top_p=1,
        stream=True,
        stop_sequences=["(TOKENSTOP)"],
        return_options=ReturnOptions(input_text=False, input_tokens=True),
        moderations=ModerationsOptions(
            # Threshold is set to very low level to flag everything (testing purposes)
            # or set to True to enable HAP with default settings
            hap=HAPOptions(input=True, output=False, threshold=0.01)
        ),
    ),
)

import ray
ray.init()

@ray.remote
def generate_response(text):
    result = llm.generate(
        prompts=[
            f"System Prompt: {SystemPrompt} \n\n The industrial asset name is {text}.",
        ]
    )
    return result.generations[0][0].text

refs = []
for i in range(30):
    refs.append(generate_response(instructions[i]))

parallel_returns = refs
#parallel_returns = ray.get(refs)

for i in range(len(instructions)):
    print ('Start...-----------------------------------------------------------------------------')
    print (instructions[i])
    answer = parallel_returns[i]
    print(answer)
    print(
        "-----------------------------------------------------------------------------...End"
    )