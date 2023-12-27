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

# Trained with 8k context length and fixed cache size, with a theoretical attention span of 128K tokens
# fix the resonable context length
# record the # token generated for each Question
# given a question, the context is function of similar questions controlled by context length limit (avoid k)
# thus context is determined by the limit of the
# How do we define the similarity between questions (just use simple sentence embedding approach)
# it can be as simple as blue score (just to avoid the costly)

llm = LangChainChatInterface(
    #model="meta-llama/llama-2-70b-chat",
    model="thebloke/mixtral-8x7b-instruct-v0-1-gptq",
    credentials=Credentials(api_key, api_endpoint),
    params=GenerateParams(
        decoding_method="sample",
        max_new_tokens=2000,
        min_new_tokens=200,
        temperature=0.5,
        top_k=50,
        top_p=1,
        stream=True,
        return_options=ReturnOptions(input_text=False, input_tokens=True),
        moderations=ModerationsOptions(
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

#print (result)
conversation_id = result.generations[0][0].generation_info["meta"]["conversation_id"]
print(f"New conversation with ID '{conversation_id}' has been created!")
print(f"Response: {result.generations[0][0].text}")
print(result.llm_output)
print(result.generations[0][0].generation_info['token_usage']) 

df = pd.read_csv('./genai_questions_1.csv')
instructions = df['questions'].to_list()
#print (result)
tokenized_text = result.generations[0][0].generation_info['input_tokens']
original_text = "".join(token['text'].replace('▁', ' ') for token in tokenized_text)
original_text = original_text.replace("<0x0A>", "\n")
print (original_text)

for prompt in instructions:
    result = llm.generate(
        messages=[[HumanMessage(content=prompt)]],
        #options=ChatOptions(conversation_id=conversation_id, use_conversation_parameters=True),
    )

    # prompt
    tokenized_text = result.generations[0][0].generation_info['input_tokens']
    original_text = "".join(token['text'].replace('▁', ' ') for token in tokenized_text)
    original_text = original_text.replace("<0x0A>", "\n")
    print (original_text)

    print(f"Response: {result.generations[0][0].text}")
    print(result.llm_output)

    print(result.generations[0][0].generation_info['token_usage'])
    print(result.generations[0][0].generation_info["meta"]["conversation_id"])
    break
