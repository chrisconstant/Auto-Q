import os

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from genai.credentials import Credentials
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.schemas import ChatOptions, GenerateParams, ReturnOptions
from genai.schemas.generate_params import HAPOptions, ModerationsOptions

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_endpoint = 'https://bam-api.res.ibm.com'


llm = LangChainChatInterface(
    model="meta-llama/llama-2-70b-chat",
    credentials=Credentials(api_key, api_endpoint),
    params=GenerateParams(
        decoding_method="sample",
        max_new_tokens=100,
        min_new_tokens=10,
        temperature=0.5,
        top_k=50,
        top_p=1,
        stream=True,
        return_options=ReturnOptions(input_text=False, input_tokens=True),
        moderations=ModerationsOptions(
            # Threshold is set to very low level to flag everything (testing purposes)
            # or set to True to enable HAP with default settings
            hap=HAPOptions(input=True, output=False, threshold=0.01)
        ),
    ),
)

prompt = "I like to build an anomaly model for wind turbine."
print(f"Request: {prompt}")
result = llm.generate(
    messages=[
        [
            SystemMessage(
                content="""You are a helpful, respectful and honest assistant.
Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature. If a question does not make
any sense, or is not factually coherent, explain why instead of answering something incorrectly.
If you don't know the answer to a question, please don't share false information.
""",
            ),
            HumanMessage(content=prompt),
        ]
    ],
)

print('------------------')
print ('++++++++++++')

#print (result)
conversation_id = result.generations[0][0].generation_info["meta"]["conversation_id"]
print(f"New conversation with ID '{conversation_id}' has been created!")
print(f"Response: {result.generations[0][0].text}")
print(result.llm_output)
#print(result.generations[0][0].generation_info)

print(result.generations[0][0].generation_info['token_usage']) 

# {'prompt_tokens': 154, 'completion_tokens': 100, 'total_tokens': 254, 'generated_token_count': 100, 'input_token_count': 154}


if True:

    for i in range(50):
        prompt = "What was the asset name we wanted to build anomaly model."
        #print(f"Request: {prompt}")
        result = llm.generate(
            messages=[[HumanMessage(content=prompt)]],
            options=ChatOptions(conversation_id=conversation_id, use_conversation_parameters=True),
        )
        conversation_id = result.generations[0][0].generation_info["meta"]["conversation_id"]
        #print(f"New conversation with ID '{conversation_id}' has been created!")
        #print(f"Response: {result.generations[0][0].text}")
        #print(result.llm_output)
        print(result.generations[0][0].generation_info['token_usage'])
        #print(result.generations[0][0].generation_info)

"""You act as a reliability engineer who is expert in failure modes and effect analysis (FMEA) of asset 
reliability. You task is to provide a accurate information about asset's component, subcomponent, failure mode
failure reason, failure code and degradation mechanisum along with severity and ability to detect them 
before it happen via preventive maintainance. You will be provided an enough information
about the asset class such as wind turbine, pump, oil well, etc. """


"""You act as a data scientist. Your job is to build an anomaly model using real time time series sensor data
 obtained from OT system. In order to get domain understanding of the problem you will prepare a series of 
 questions to be asked in sequential orders to subject matter experts. Typical questions should focus on the
important components for which anomaly model should be build, the important failure modes and the ability of
 sensor data to detect these failure.""" 


# reliability engineer who is expert in failure modes and effect analysis (FMEA) of asset 
# reliability. You task is to provide a accurate information about asset's component, subcomponent, failure mode
# failure reason, failure code and degradation mechanisum along with severity and ability to detect them 
# before it happen via preventive maintainance. You will be provided an enough information
# about the asset class such as wind turbine, pump, oil well, etc. 