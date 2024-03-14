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
api_key = "pak-KnqohFjgmJKou_eirLnIJMtbxdFzUYylnzScnLxVhY0"
api_endpoint = "https://bam-api.res.ibm.com"

SystemPrompt = """
You are a helpful, respectful, and honest assistant. You will be introduced to several 
question types such as Verification, Disjunctive, Concept, etc. The user will provide a 
question and you will recommed a question type using the provided annotation guideline. The
definition for each question type is given as follows, along with examples per question type. Here are 
the annotation guidelines for ten question types along with their description.  

1. Question Type: Verification 
Description: A question of this type asks for the truthfulness of an event or a concept.

2. Question Type: Disjunctive 
Description: A question of this type asks for the true one given multiple events or concepts, 
where comparison among options is not needed.

3. Question Type: Concept 
Description: A question of this type asks for a definition of an event or a concept.

4. Question Type: Extent 
Description: A question of this type asks for the extent or quantity of an event or a concept.

5. Question Type: Example 
Description: A question of this type asks for example(s) or instance(s) of an event or a concept.

6. Question Type: Comparision 
Description: A question of this type asks for comparison among multiple events or concept.

7. Question Type: Cause 
Description: A question of this type asks for the cause or reason for an event or a concept.

8. Question Type: Consequence 
Description: A question of this type asks for the consequences or results of an event.

9. Question Type: Procedural 
Description: A question of this type asks for the procedures, tools, or methods by which a certain outcome is achieved.

10. Question Type: Judgmental
Description: A question of this type asks for the opinions of the answerer's own.

"""

ClassifierPrompt = """
Assuming the user is seeking guidance about the type of the given question. There is a possibility
that more than one question types can be associated with a given question. Your answer should include 
all question types.

Question: Is there any historical data on gearbox failures or anomalies that can be used for model training? Please use (Internal thought).

(Internal thought): first we find out the list of candidate question types mentioned in System Prompt. 

We found ten question types listed in system prompt: [Verification, Disjunctive, Concept, Extent, Example, Comparision, Cause, Consequence, Procedural, Judgmental]. 

First, let us evaluate first question type (Verification). A Verification question asks for the truthfulness of an event or concept. 
The given question does not ask for the truthfulness of a specific event or concept, but rather asks for information about 
the existence of historical data related to gearbox failures or anomalies. The sentiment for Verification is negative.

Second, let us evaluate second question type (Disjunctive). A Disjunctive question asks for the true one given multiple events or 
concepts, where comparison among options is not needed. The given question does not present multiple events or concepts that 
require selection, but rather asks for information about a specific topic. The sentiment for Disjunctive is negative.

Third, let us evaluate third question type (Concept). The question asks for information about the existence of historical data related to 
gearbox failures or anomalies, which aligns with the definition of a Concept question. Concept questions ask for a definition or 
explanation of a concept or event. The sentiment for Concept is positive.

Fourth, let us evaluate forth question type (Extent). An Extent question asks for the extent or quantity of an event or concept. 
The given question does not ask for the extent or quantity of gearbox failures or anomalies, but rather asks for historical data related 
to these events. The sentiment for Extent is negative.

Fifth, let us evaluate fifth question type (Example). An Example question asks for example(s) or instance(s) of an event or concept. 
The given question does not ask for specific examples of gearbox failures or anomalies, but rather asks for historical data related to 
these events. The sentiment for Example is negative.

Sixth, let us evaluate sixth question type (Comparison). A Comparision question asks for comparison among multiple events or concepts. 
The given question does not ask for comparison among different types of gearbox failures or anomalies, but rather asks for historical 
data related to these events. The sentiment for Comparison is negative.

Seventh, let us evaluate seventh question type (Cause). A Cause question asks for the cause or reason for an event or concept. 
The given question does not ask for the cause of gearbox failures or anomalies, but rather asks for historical data related 
to these events. The sentiment for Cause is negative.

Eighth, let us evaluate eighth question type (Consequence). A Consequence question asks for the consequences or results of an event. 
The given question does not ask for the consequences of gearbox failures or anomalies, but rather asks for historical data related 
to these events.The question is not asking for the consequences or results of an event. The sentiment for Consequence is negative.

Ninth, let us evaluate ninth question type (Procedural). A Procedural question asks for the procedures, tools, or methods by which a 
certain outcome is achieved. The given question does not ask for procedures or methods related to gearbox failures or anomalies, 
but rather asks for historical data related to these events, The sentiment for Procedural is negative.

Tenth, let us evaluate tenth question type (Judgmental). A Judgmental question asks for the opinions of the answerer's own. 
The given question does not ask for the opinion of the answerer, but rather asks for factual information about historical data 
related to gearbox failures or anomalies. The sentiment for Judgmental is negative. 

Overall, Concept has positive sentiment.

Answer: The final answer is Concept. (TOKENSTOP)

"""

extraQ = """Question: Is Michael Jackson an African American?
Answer: Verification

Question: Does a Mercedes dealer have to unlock a locked radio?
Answer: Verification

Question: Is Michael Jackson an African American or Latino?
Answer: Disjunctive

Question: “Is a DVI to HDMI cable supposed to transmit audio and video or just video?
Answer: Disjunctive

Question: What is the origin of the phrase "kicking the bucket"?
Answer: Concept

Question: How long does gum stay in your system?
Answer: Extent

Question: What is Barry Larkin's hat size?
Answer: Extent

Question: What are some examples to support or contradict this?
Answer: Example

Question: What countries/regions throughout the world do not celebrate the Christmas holidays?
Answer: Example
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

import pandas as pd

df = pd.read_csv("../../results/windturbinegearbox/genai_questions_bank_windturbinegearbox_llama.csv")
instructions = df["questions"].to_list()

llm = LangChainInterface(
    model=LLMsets[3],
    credentials=Credentials(api_key, api_endpoint),
    params=GenerateParams(
        decoding_method="greedy",
        max_new_tokens=2000,
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

import ray

ray.init()


@ray.remote
def generate_response(text):
    result = llm.generate(
        prompts=[
            f"System Prompt: {SystemPrompt} \n\n {ClassifierPrompt} \n\n Question: {text} Please use (Internal thought)."
        ]
    )
    return result.generations[0][0].text


refs = []
for i in range(len(instructions)):
    refs.append(generate_response.remote(instructions[i]))

parallel_returns = ray.get(refs)
final_answer = []
for i in range(len(instructions)):
    sindex = parallel_returns[i].rfind("Answer:")
    eindex = parallel_returns[i].rfind("(TOKENSTOP")
    answer_text = parallel_returns[i][sindex + 7 : eindex]
    final_answer.append(answer_text)

for i in range(len(instructions)):
    print(
        "Start...-----------------------------------------------------------------------------"
    )
    print(instructions[i])
    answer = final_answer[i]
    print(answer)
    print(
        "-----------------------------------------------------------------------------...End"
    )

df = pd.DataFrame([instructions,final_answer])
df = df.T
df.columns = ['instruction', 'final_answer']
df.to_csv('question_type_classification_answer_windturbine_lamma.csv',index=False)
