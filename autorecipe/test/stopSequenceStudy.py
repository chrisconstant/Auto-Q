import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import pandas as pd
from genai.credentials import Credentials
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.schemas import ChatOptions, GenerateParams, ReturnOptions
from genai.schemas.generate_params import HAPOptions, ModerationsOptions

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_endpoint = 'https://bam-api.res.ibm.com'

llm = LangChainChatInterface(
    #model="meta-llama/llama-2-70b-chat",
    model="thebloke/mixtral-8x7b-instruct-v0-1-gptq",
    credentials=Credentials(api_key, api_endpoint),
    params=GenerateParams(
        decoding_method="greedy",
        max_new_tokens=2000,
        min_new_tokens=10,
        temperature=0.5,
        top_k=50,
        top_p=1,
        stream=True,
        stop_sequences = ["(TOKENSTOP)"],
        return_options=ReturnOptions(input_text=False, input_tokens=True),
        moderations=ModerationsOptions(
            hap=HAPOptions(input=True, output=False, threshold=0.01)
        ),
    ),
)

print (llm)
print (llm.params.stop_sequences)

prompt = """
1. What is the typical failure rate of wind turbine gearboxes and what are the common modes of failure for this type of equipment?
2. What are the key performance indicators (KPIs) for a wind turbine gearbox that we should focus on for anomaly detection, such as vibration, temperature, or sound levels?
3. Are there any specific threshold values for these KPIs that indicate a potential failure or abnormal behavior?
4. How is the sensor data currently collected and processed, and what is the sampling frequency of the data?
5. Are there any known issues with sensor data quality, such as missing data, data outliers, or sensor drift, that could impact the performance of the anomaly detection model?
6. Have any past anomalies or failures been recorded, and if so, what was the root cause and how was it detected?
7. Are there any specific environmental or operational conditions that are known to increase the likelihood of gearbox failures, such as extreme temperatures, high wind speeds, or variable loads?
8. What are the consequences of a gearbox failure, such as downtime, repair costs, or safety risks, and how important is it to detect and mitigate these failures as early as possible?
"""

llm.params.min_new_tokens = 100

result = llm.generate(
    messages=[
        [
            SystemMessage(
                content="""
Pretend you are a question generation system. I will give you a list of questions or a pair of question and answer 
extracted from the conversation between two users where question is asked by data scientist and 
subject matter expert has provided corresponding answer. Based on the conversation, you reply me with additional set of 
questions that data scientist can ask to subject matter expert. The newly generated questions must align with original set of questions. 
you should avoid generating duplicate questions. you should also avoid questions for which potential answer can be similar. 
Please do not use a conversational approach to ask questions and gather information.
""",
            ),
            HumanMessage(content=prompt),
        ]
    ],
    stop=["11. "],
)

print(f"Response: {result.generations[0][0].text}")

