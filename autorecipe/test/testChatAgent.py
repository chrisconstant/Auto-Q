import os
from dotenv import load_dotenv

from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()
api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_endpoint = "https://bam-api.res.ibm.com"
creds = Credentials(api_key, api_endpoint=api_endpoint)

print("\n------------- Example (LangChain)-------------\n")

# params = GenerateParams(decoding_method="greedy",  max_new_tokens=4096)
params = GenerateParams(
    decoding_method="greedy",
    max_new_tokens=3000,
    min_new_tokens=200,
    temperature=0.5,
    top_k=50,
    top_p=1,
    stream=True,
    stop_sequences=["Question"],
)

print("Using GenAI Model expressed as LangChain Model via LangChainInterface:")

modelSet = [
    "meta-llama/llama-2-70b-chat",
    "thebloke/mixtral-8x7b-instruct-v0-1-gptq",
]

langchain_model = LangChainInterface(
    model=modelSet[1], params=params, credentials=creds
)

background_ds = """Your job is to build an anomaly model using real time time series sensor data
 obtained from IoT/OT system. In order to get domain understanding of the problem you will prepare a series of 
 questions to be asked in sequential orders to subject matter experts. Typical questions should focus on the
important components for which anomaly model should be build, the important failure modes and the ability of
 sensor data to detect these failure. """

ans = langchain_model(background_ds + "\n Answer this question: Generate 100 questions for building anomaly detection model for wind turbine gearbox")
print (ans)