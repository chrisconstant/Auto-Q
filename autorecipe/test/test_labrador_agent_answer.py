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

SMESystemPrompt = """
    You act as a subject matter expert who is expert in failure modes and effect analysis (FMEA) of asset or system 
    reliability. 
    
    Here is FMEA Procedure:

    An FMEA is a systematic method of identifying and preventing product and process reliability 
    problems before they occur. FMEA focus on preventing defects, improving safety and reliability, and increasing 
    customer satisfaction. The method does not require complicated statistics only simple arithmetic. 
    FMEA considers each failure mode of every component from the least up to the greatest. Your task is to provide 
    accurate information about asset's component, subcomponent, assembly, failure mode, failure cause, failure code 
    and degradation mechanism, degradation influence along with severity, likelihood and detectability of the point 
    of failure (P-F Curve) so immediate intervention in terms of maintenance or operations can be taken to extend 
    the life of the asset and or system. It determines a risk priority number for ranking efforts. it is a bottom-up
    approach for worst case estimates in a as search for effects of an item failure on operation of the system. 
    FMEA predicts potential problems, identifies possible causes, assesses effects and helps plan preemptive 
    corrective action. 
    
    FMEA considers each failure mode of every component from the least up to the greatest. 
    It determines a risk priority number for ranking efforts. FMEA predicts potential problems, identifies 
    possible causes, assesses effects, and helps plan preemptive corrective action. The Reliability Engineer 
    crafts a strategy of both preventive and condition-monitoring tasks that are specifically designed to 
    identify or prevent failure modes. By using collected information about the equipment, failure modes are 
    identified, and the appropriate tasks are selected to identify or prevent these failure modes as 
    early as possible. The strategy is implemented according to the criticality database. 
    
    The relative risk of a failure and its effects is determined by three factors. 
    1. Severity: The consequence of the failure should it occur, 
    2. Occurrence: The probability or frequency of the failure occurring and 
    3. Detection: The probability of the failure being detected before the impact and consequence of the 
    failure is realized. 
    
    Using the data and the knowledge of the process or product, each potential 
    failure mode and effect is rated in each of these factors on a scale ranging from 1 to 10. 
    By multiplying the ranking for the three factors (Severity x Occurrence x Detection), 
    a risk priority number will be determined for each potential failure mode and effect. 
    The RPN (which will range from 1 to 1000 for each failure mode) us used to rank the need for 
    corrective actions to eliminate or reduce the potential failure mode. 

    User will provide a asset class or ask FMEA related question and you will provide descriptive FMEA 
    information by following above FMEA Procedure. 
    """

MSMESystemPrompt = """
    You act as a subject matter expert who is expert in failure modes and effect analysis (FMEA) of asset or system 
    reliability. 
    
    Here is FMEA Procedure:

    An FMEA is a systematic method of identifying and preventing product and process reliability 
    problems before they occur. FMEA focus on preventing defects, improving safety and reliability, and increasing 
    customer satisfaction. The method does not require complicated statistics only simple arithmetic. 
    FMEA considers each failure mode of every component from the least up to the greatest. Your task is to provide 
    accurate information about asset's component, subcomponent, assembly, failure mode, failure cause, failure code 
    and degradation mechanism, degradation influence along with severity, likelihood and detectability of the point 
    of failure (P-F Curve) so immediate intervention in terms of maintenance or operations can be taken to extend 
    the life of the asset and or system. It determines a risk priority number for ranking efforts. it is a bottom-up
    approach for worst case estimates in a as search for effects of an item failure on operation of the system. 
    FMEA predicts potential problems, identifies possible causes, assesses effects and helps plan preemptive 
    corrective action. 
    
    FMEA considers each failure mode of every component from the least up to the greatest. 
    It determines a risk priority number for ranking efforts. FMEA predicts potential problems, identifies 
    possible causes, assesses effects, and helps plan preemptive corrective action. The Reliability Engineer 
    crafts a strategy of both preventive and condition-monitoring tasks that are specifically designed to 
    identify or prevent failure modes. By using collected information about the equipment, failure modes are 
    identified, and the appropriate tasks are selected to identify or prevent these failure modes as 
    early as possible. The strategy is implemented according to the criticality database. 
    
    The relative risk of a failure and its effects is determined by three factors. 
    1. Severity: The consequence of the failure should it occur, 
    2. Occurrence: The probability or frequency of the failure occurring and 
    3. Detection: The probability of the failure being detected before the impact and consequence of the 
    failure is realized. 
    
    Using the data and the knowledge of the process or product, each potential 
    failure mode and effect is rated in each of these factors on a scale ranging from 1 to 10. 
    By multiplying the ranking for the three factors (Severity x Occurrence x Detection), 
    a risk priority number will be determined for each potential failure mode and effect. 
    The RPN (which will range from 1 to 1000 for each failure mode) us used to rank the need for 
    corrective actions to eliminate or reduce the potential failure mode. 

    User will provide a asset class or ask FMEA related question and you will provide descriptive FMEA 
    information by following above FMEA Procedure. 
    """


InfoExtractor = """
You are helpful AI Assistant. You will be given an input passage and a question. You will carefully read 
the passage and produce a reliable answer only using information given in passage. Do not provide any
description for the answer. If you do not find answer, then say I do not know.   
"""

FailureEffectQuestion = """
The industrial asset class is Electrical Submersible Pump. What are the failure effect of each failure mode of Electrical Submersible Pump in 
given passage? Only return the answer in the format of a numbered list. 
"""

FailureCauseQuestion = """
The industrial asset class is Electrical Submersible Pump. What are the failure causes of each failure mode of Electrical Submersible Pump in 
given passage? Only return the answer in the format of a numbered list.
"""

FailureModeQuestion = """
The industrial asset class is Electrical Submersible Pump. Find the component of Electrical Submersible Pump in given 
passage. Only return the answer in the format of a numbered list.
"""

ComponentQuestion = """
The industrial asset class is Electrical Submersible Pump. What are the component of Electrical Submersible Pump in 
given passage? Only return the answer in the format of a numbered list. If you do not find answer, then say I do not know.
"""

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "google/flan-ul2",
    "thebloke/mixtral-8x7b-instruct-v0-1-gptq",
    "ibm/granite-13b-labrador-rc",
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
        stop_sequences=["(TOKENSTOP)","User:","USER:","|user|"],
        return_options=ReturnOptions(input_text=False, input_tokens=True),
        moderations=ModerationsOptions(
            # Threshold is set to very low level to flag everything (testing purposes)
            # or set to True to enable HAP with default settings
            hap=HAPOptions(input=True, output=False, threshold=0.01)
        ),
    ),
)

import pandas as pd
df = pd.read_csv('genai_questions_answer_sme_bank_ElectricalSubmersiblePump_granite.csv')
passages = list(df['answers'])

info = '\n\n'.join(passages)
info = "Here is the passage: \n" + info

SummaryMessage = """
Please help me to generate FMEA documentation into a markdown format using the following passage. 
If needed, you can use the table format inside of the markdown file. The output has two parts:
1. It is the beginning part of the document. Please briefly introduce 
the ESP, its components and subcomponents;
2. The body part include the failure mode failure cause and failure impact of the 
ESP based on given information.        
"""

print (info)

result = llm.generate(prompts=[f"System Prompt: {SummaryMessage}  \n\n {info}"]
)
print('---------------------')
print (result.generations[0][0].text)