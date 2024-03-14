from genai.extensions.langchain import LangChainInterface
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.credentials import Credentials
import uuid
import mlflow

from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationReturnOptions,
    TextGenerationParameters,
)
from autorecipe.genai.GenAIInstruct import GenAIInstructClient
import re
import json

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
credentials = {
    "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
    "api_endpoint": "https://bam-api.res.ibm.com",
}

asset_class = "Electrical Submersible Pump"
asset_description = "An electrical submersible pump (ESP) is a type of pump used to extract oil or gas from underground reservoirs. It is a submersible pump that is designed to operate while submerged in the fluid. The ESP consists of several components, including a power section, a pump section, a cable, and sucker rods. The power section contains the motor, which converts electrical energy into mechanical energy. The pump section contains the pump, which is used to lift the fluid to the surface. The cable connects the power section to the pump section and carries the electrical energy to the pump. The sucker rods are used to connect the pump to the wellbore and are typically made of steel. The main components of an electrical submersible pump include the power section, the pump section, the cable, and the sucker rods. The power section typically includes a squirrel cage induction motor, which is designed to operate efficiently in high-torque, low-speed applications. The pump section typically includes a centrifugal pump, which is used to lift the fluid to the surface. The cable is typically made of high-voltage, low-resistance cable, which is designed to minimize power loss and maximize efficiency. The sucker rods are used to connect the pump to the wellbore and are typically made of steel."
passage = "Your Passage Associated with the Asset"

MultipleAnswerTemplate = """
I am building FMEA documentation. I have provided an asset class and asset description. I have collected multiple 
answers that includes components, subcomponents and assembly of asset {asset_class}. You need to generate 
a table that aims to provide a comprehensive overview of the {asset_class}'s components and subcomponents, 
assemblies, and additional components based on the collected answers.

Asset Class: {asset_class}

Passage: {passage} 

Your task is to combine all these answers and generate the final unified answer. Your answer should contain 
the components and subcomponents and assembly of asset {asset_class} using information provided in passage.
Generate the output in a form of markdown file with table with three columns: section, component or subcomponent and assembly.
In the markdown file, replace the "Not Specified" entries with "N/A" if no information is available.
"""

FailureAnswerTemplate = """
I am building FMEA documentation. I have provided an asset class and asset description. I have collected multiple 
answers that includes failure modes, failure causes and failure effects for an {asset_class}. You need to generate 
a table that aims to provide a comprehensive overview of the {asset_class}'s failure modes, failure causes and 
failure effects for component wise. You can also include failure information at subcomponents, assemblies, and 
additional components if it is available. You should generate information only based on the collected answers.

Asset Class: {asset_class}

Passage: {passage} 

Your task is to combine all these answers and generate the final unified answer. Your answer should contain 
the failure modes, failure causes and failure effects of an asset {asset_class} using information provided in passage.
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

params = {
    "decoding_method": DecodingMethod.GREEDY,
    "min_new_tokens": 200,
    "max_new_tokens": 2000,  # 1500,
    "stop_sequences": ["(TOKENSTOP)","User:","USER:","Assistant:","ASSISTANT:"],
    "return_options": TextGenerationReturnOptions(input_text=False, input_tokens=True),
    "moderations": ModerationParameters(
        hap=ModerationHAP(input=True, output=False, threshold=0.01)
    ),
}

with open('component_list.json', 'r') as file:
    answers = json.load(file)

# print (answers)
joined_answers = "\n".join([f"Answer {index + 1}. {answer}" for index, answer in enumerate(answers)])

components_ans = []
failure_loc_ans = []

filled_template = MultipleAnswerTemplate.format(
    asset_class=asset_class, passage=joined_answers
)

experiment_name = (
    "MyExperiment_" + asset_class + "_" + str(uuid.uuid4())
)
experiment_id = mlflow.create_experiment(experiment_name)

llm = GenAIInstructClient(name='AIClient',
                description='Test',
                skill='AAA',
                model=LLMsets[3],
                credentials=credentials,
                system_message=filled_template,
                question_message=None,
                stream=False,
                params=params,
                )

result = llm.create(context=None, messages='', experiment_id=experiment_id)
print (joined_answers)

result = result.split("I do not know")[0]
result = result.split("Note: The ")[0]
result = result.split("Please note ")[0]
components_ans.append(result)
print("---------------------")
# print (passage)
print(result)
#break

