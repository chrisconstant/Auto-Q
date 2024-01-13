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
You are a helpful, respectful, and honest assistant. User will provide a 
question and you will annotate the question as useful for learning when all of the following
conditions hold:

1. the question is concept-relevant
2. the question is context-complete
3. the question is not generic

"""

ClassifierPrompt = """
Assuming user is seeking guidance about usefulness of given question. You will use the example provided 
in a form of Internal thought to find the answer for all questions.

Question: What is the typical failure rate and common failure modes of wind turbine gearboxes? Please use (Internal thought).

(Internal thought): first we find out the list of conditions mentioned in System Prompt. 

We found three conditions listed in system prompt: [concept-relevant, context-complete, non-generic]. 

First, the question is directly related to the concept of wind turbine gearboxes and their performance, 
which is a crucial aspect of wind energy systems. Therefore, it is concept-relevant. 
The sentiment for concept-relevant is positive.

Second, The question provides sufficient context by specifying the component (wind turbine gearboxes) and 
the desired information (typical failure rate and common failure modes). This information is necessary to
 provide a meaningful answer and to help the learner understand the topic better. Therefore, it is context-complete.
 The sentiment for concept-complete is positive.

Third, The question is specific to wind turbine gearboxes and their unique challenges, rather than asking
 a generic question about gearboxes or mechanical systems in general. This specificity ensures that the 
 answer will provide valuable insights into the particular topic, making it non-generic. 
 The sentiment for non-generic is positive.

Overall, all conditions have positive sentiment. Hence, the question is useful for learning.

Answer: The final answer is useful. (TOKENSTOP)
"""

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "google/flan-ul2",
    "thebloke/mixtral-8x7b-instruct-v0-1-gptq",
]

# Is this question for Subject Matter Expert?
# Is this question for Data Scientist?

import pandas as pd

df = pd.read_csv("./genai_questions.csv")
df = pd.read_csv(
    "../question_classification/question_classifier/data/test.csv",
    sep="\t",
    header=None,
)
df.columns = ["Source"]
df["label"] = df["Source"].apply(lambda x: x[0])
df["questions"] = df["Source"].apply(lambda x: x[2:])
df = df[["label", "questions"]]
print(df)

instructions = df["questions"].to_list()

llm = LangChainInterface(
    model=LLMsets[1],
    credentials=Credentials(api_key, api_endpoint),
    params=GenerateParams(
        decoding_method="greedy",
        max_new_tokens=500,
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

ansSet = []
for qindex, qpromt in enumerate(instructions):
    print(
        "Start...-----------------------------------------------------------------------------"
    )
    print(qpromt)
    result = llm.generate(
        prompts=[
            f"System Prompt: {SystemPrompt} \n\n {ClassifierPrompt} \n\n Question: {qpromt}. Please use (Internal thought)."
        ]
    )
    answer = result.generations[0][0].text
    print (answer)
    sindex = answer.rfind("Answer:")
    eindex = answer.rfind("(TOKENSTOP")
    print(answer[sindex + 7 : eindex])  # 7 = len('Answer:')
    labelX = "0"
    if "is useful" in answer[sindex + 7 : eindex]:
        labelX = "1"
    print (labelX)
    ansSet.append(labelX)
    print(
        "-----------------------------------------------------------------------------...End"
    )

if len(ansSet) == len(instructions):
    df["answer"] = ansSet
    df.to_csv(
        "../question_classification/question_classifier/data/testResult_lamma.csv",
        index=False,
    )
