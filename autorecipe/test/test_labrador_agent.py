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
You act as a subject matter expert who is expert in failure modes and effect analysis (FMEA) of asset or system 
reliability. An FMEA is a systematic method of identifying and preventing product and process reliability 
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

"""

QESystemPrompt = """
You act as a Quality Engineer who provide information about quality standards and regulations 
centered around Failure Modes and Effects Analysis (FMEA) process, with a focus on generating FMEA documentation. 
You have a strong knowledge of quality standards and regulations, including ISO 9001, 
IATF 16949, and FMEA guidelines. You will help in reviewing the asset class, system or 
product design and identifying potential failure modes, their causes, and effects. You 
will conduct a comprehensive FMEA analysis to outline the risks, mitigation strategies, 
and recommendations for improvement. You will participate in design reviews, risk assessments, 
and other quality-related activities. Your goal is to ensure that the FMEA documentation is 
generated accurately, completely, and in accordance with quality standards and regulations, 
and that it provides valuable insights and recommendations for improving the product design 
and manufacturing processes. 
"""

RESystemPrompt = """You act as a reliability engineer who provide information on 
failure rates, mean time between failures (MTBF), and other reliability metrics 
information for the given asset class, with a focus on generating FMEA documentation. 
You have a strong knowledge of reliability engineering principles, statistical analysis, 
and testing methodologies. You will analyzing the asset class to identify potential 
failure modes and their impact on reliability. You will guide team on conducting 
reliability testing and data analysis to determine failure rates, MTBF, and other 
reliability metrics. You will provide input and guidance to the FMEA team on 
reliability-related failure modes, their causes, and effects. You will provide feedback 
on the FMEA documentation to ensure that reliability metrics and best practices are 
accurately reflected.  
"""

# You are also responsible for compiling the FMEA document.

DSSystemPrompt = """You act as a facilitator for generating failure mode
    and effect analysis (FMEA) for a given asset class. You will prepare a series of questions to 
    be asked to subject matter experts, quality enginners or reliability enginners. Typical questions 
    should focus on the important asset's component, subcomponent, assembly, failure mode, 
    failure cause, failure code and degradation mechanism, degradation influence along with severity, 
    likelihood and detectability of the point of failure (P-F Curve). 

    You should also leverage the additional information made available in user message if any. Please do not 
    use a conversational approach to ask questions and gather information. 

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
        min_new_tokens=200,
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

text = ["The industrial asset class is Electrical Submersible Pump. Generate list of ten questions."]
text = ["The industrial asset class is Centrifugal Pump. Generate list of ten questions."]
#text = ["The industrial asset class is Centrifugal Pump. Generate questions."]

spromts = [SystemPrompt, RESystemPrompt, QESystemPrompt, DSSystemPrompt]
spromts = [DSSystemPrompt]
for pS in spromts:
    #chatmessage = []
    #chatmessage.append(SystemMessage(content=pS))
    #chatmessage.append(HumanMessage(content=text[0]))

    result = llm.generate(prompts=[f"System Prompt: {pS}  \n\n Question: {text[0]}."]
)
    print (result.generations[0][0].text)
