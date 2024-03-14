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


asset_class = "Electrical Submersible Pump"
asset_description = "An electrical submersible pump (ESP) is a type of pump used to extract oil or gas from underground reservoirs. It is a submersible pump that is designed to operate while submerged in the fluid. The ESP consists of several components, including a power section, a pump section, a cable, and sucker rods. The power section contains the motor, which converts electrical energy into mechanical energy. The pump section contains the pump, which is used to lift the fluid to the surface. The cable connects the power section to the pump section and carries the electrical energy to the pump. The sucker rods are used to connect the pump to the wellbore and are typically made of steel. The main components of an electrical submersible pump include the power section, the pump section, the cable, and the sucker rods. The power section typically includes a squirrel cage induction motor, which is designed to operate efficiently in high-torque, low-speed applications. The pump section typically includes a centrifugal pump, which is used to lift the fluid to the surface. The cable is typically made of high-voltage, low-resistance cable, which is designed to minimize power loss and maximize efficiency. The sucker rods are used to connect the pump to the wellbore and are typically made of steel."
passage = "Your Passage Associated with the Asset"

FailureAnswerTemplate = """
I am building FMEA documentation. I have provided an asset class and asset description. I have collected multiple 
answers that includes failure modes, failure causes and failure effects for an {asset_class}. You need to generate 
a table that aims to provide a comprehensive overview of the {asset_class}'s failure modes, failure causes and 
failure effects for component wise. You can also include failure information at subcomponents, assemblies, and 
additional components if it is available. You should generate information only based on the collected answers that 
are provided in Passage below.

Asset Class: {asset_class}

Passage: {passage} 

Your task is to combine all these answers and generate the final unified answer. Your answer should contain 
the component/subcomponent wise failure modes, failure causes and failure effects of an asset {asset_class} using information provided in passage.
Generate the output in a form of markdown file with table with four columns: component or subcomponent, 
failure modes, failure causes and failure effects. In the markdown file, replace the "Not Specified" entries with "N/A" if no information is available.
"""

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v0-1-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
]

# Is this question for Subject Matter Expert?
# Is this question for Data Scientist?

llm = LangChainInterface(
    model=LLMsets[4],
    credentials=Credentials(api_key, api_endpoint),
    params=GenerateParams(
        decoding_method="greedy",
        max_new_tokens=2000,
        min_new_tokens=10,
        temperature=0.5,
        top_k=50,
        top_p=1,
        stream=True,
        stop_sequences=["(TOKENSTOP)", "User:", "USER:", "|user|"],
        return_options=ReturnOptions(input_text=False, input_tokens=True),
        moderations=ModerationsOptions(
            # Threshold is set to very low level to flag everything (testing purposes)
            # or set to True to enable HAP with default settings
            hap=HAPOptions(input=True, output=False, threshold=0.01)
        ),
    ),
)

import pandas as pd

df = pd.read_csv(
    "failure_list.csv"
)
answers = list(df["failures"])
joined_answers = "\n".join([f"Answer {index + 1}. {answer}" for index, answer in enumerate(answers)])

components_ans = []
failure_loc_ans = []

filled_template = FailureAnswerTemplate.format(
    asset_class=asset_class, passage=joined_answers
)

print (filled_template)

result = llm.generate(prompts=[f"System Prompt: {filled_template}"])
result = result.generations[0][0].text
print (result)
result = result.split("I do not know")[0]
result = result.split("Note: The ")[0]
result = result.split("Please note ")[0]
components_ans.append(result)
print("---------------------")
# print (passage)
print(result)
#break

