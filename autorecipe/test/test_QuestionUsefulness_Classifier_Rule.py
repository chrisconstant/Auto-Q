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
You are a helpful, respectful, and honest assistant. User will provide list of questions 
written by various different persona. These questions are labelled as useful question. Your role is
to provide rules that justify why these questions are labelled as useful.   

Here are some examples of guideline that define what is usefulness of question means. 
1. Questions that are relevant, informative, and thought-provoking in the context of particular domain 
are candidates for "usefulness". 
2. Question is useful if it has a level of depth and complexity that makes them useful for learning and discussion. 
3. The questions are clear and concise, making it easy for others to understand and respond to them.
Note that these guidelines are not complete and you should suggest any other guidelines.

"""

ClassifierPrompt = """

Here is a list of questions labelled as "useful":
- where would one find out the tax bracket rates for their country ?
- hey , is n't the price and hence demand predetermined by the industry ?
- what about 'investing ' in a ponzi scheme and withdrawing in the early days before it busts ?
- when would the value of a product increase ?
- at 8:50 when grandmother deposited the cash in the banks a/c , why did only the reserve rise , why not the liabilities and hence the total balance sheets of the bank ?
- a person in the market who expect that the prices of products will increase in future is known as ?
- why do people from forex always advertise and want people to exchange what benefits do they gain ?
- can a bank be a member of more than one network ?
- if i have a 15 year fixed mortgage , it means i have a 15 year amortization ?
- do the fixed rates adjust to inflation over their periods ?

Can you generate some rule on why these questions are labelled as useful?

"""


LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "google/flan-ul2",
    "thebloke/mixtral-8x7b-instruct-v0-1-gptq",
]

# Is this question for Subject Matter Expert?
# Is this question for Data Scientist?

for i in range(4):

    llm = LangChainInterface(
        model=LLMsets[i],
        credentials=Credentials(api_key, api_endpoint),
        params=GenerateParams(
            decoding_method="greedy",
            max_new_tokens=1000,
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

    print(
        "Start...-----------------------------------------------------------------------------"
    )
    answer = llm.generate(
        prompts=[
            f"System Prompt: {SystemPrompt} \n\n {ClassifierPrompt} ."
        ]
    )
    print(answer.generations[0][0].text)
    print(
        "-----------------------------------------------------------------------------...End"
    )
