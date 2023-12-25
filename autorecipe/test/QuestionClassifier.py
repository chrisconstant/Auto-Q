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

QuestionClassifier = """
You are a helpful, respectful, and honest assistant. You will be introduced to several 
persona such as data scientists, subject matter experts, narrators etc. User will provide a 
question and you will select a persona who can answer the given question. Your selection is based 
on the persona's field experience and scientific knowledge. Sometime questions can be answered by 
multiple personas. Here is the two personas along with their skill description.  

Persona: Data Scientist
Skill: building machine learning model, data analytics, python programming

Persona: Subject Matter Expert
Skill: Provide domain knowledge for a particular industrial assets and their working condition

Persona: Testing
Skill: Provide domain knowledge for a particular industrial assets and their working condition
"""

QuestionClassifierPropmt = """
Assuming user is seeking guidance about who can answer the following question. There is a possibility
that more than one personas can provide different level of information. Your answer should include 
all personas who can be the best persona to consult for the question.

Question: 

"""

QuestionClassifierPropmt1 = """
Assuming user is seeking guidance about who can answer the following question. There is a possibility
that more than one personas can provide different level of information. Your answer should include 
all personas who can be the best persona to consult for the question. You will use the example provided 
in a form of Internal thought to find the answer for all questions.

Question: What are the most common failure modes for wind turbine gearboxes? This will help me identify the failure modes that the anomaly model should be able to detect. 
Please use (Internal thought).

(Internal thought): first we find out the list of candidate personas mentioned in system prompt. 

We found two personas listed in system prompt: [Subject Matter Expert, Data Scientist]. 

First, let us evaluate first persona (Subject Matter Expert). The Subject Matter Expert has knowledge of the domain and can provide
 information about the common failure modes for wind turbine gearboxes. This question is best suited for a Subject Matter Expert.
 The sentiment for Subject Matter Expert is positive.

Next, we evaluate second persona (Data Scientist). This question is primarily related to the domain knowledge of
 wind turbine gearboxes and their failure modes. Therefore, Data Scientist is not the best persona to consult for this question. 
 The sentiment for Data Scientist is negative. 

Overall, Subject Matter Expert has positive sentiment.

Answer: The final answer is Subject Matter Expert. 

"""

#Is this question for Subject Matter Expert?
#Is this question for Data Scientist?

import pandas as pd
df = pd.read_csv('./genai_questions_1.csv')
instructions = df['questions'].to_list()

llm = LangChainChatInterface(
    model="meta-llama/llama-2-70b-chat",
    credentials=Credentials(api_key, api_endpoint),
    params=GenerateParams(
        decoding_method="greedy",
        max_new_tokens=500,
        min_new_tokens=200,
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

for qindex, qpromt in enumerate(instructions):
    print(f"Question: {qpromt} Please use (Internal thought).")
    result = llm.generate(
        messages=[
            [
                SystemMessage(
                    content=QuestionClassifier,
                ),
                HumanMessage(content=QuestionClassifierPropmt1 + '\n' + qpromt),
            ]
        ],
    )
    print (result.generations[0][0].text)
    if qindex > 15:
        break
    print ('-----------------------------------------------------------------------------')


'''
for qindex, qpromt in enumerate(instructions):
    print(f"Question: {qpromt}")
    result = llm.generate(
        messages=[
            [
                SystemMessage(
                    content=QuestionClassifier,
                ),
                HumanMessage(content=QuestionClassifierPropmt + qpromt + ' \n Is this question for Data Scientist?'),
            ]
        ],
    )
    print (result.generations[0][0].text)

    print(f"Question: {qpromt}")
    result = llm.generate(
        messages=[
            [
                SystemMessage(
                    content=QuestionClassifier,
                ),
                HumanMessage(content=QuestionClassifierPropmt + qpromt + ' \n Is this question for Subject Matter Expert?'),
            ]
        ],
    )
    print (result.generations[0][0].text)



    if qindex > 3:
        break

'''