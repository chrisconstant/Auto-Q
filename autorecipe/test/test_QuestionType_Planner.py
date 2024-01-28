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
Your task is to come up with a short plan to help me accomplish my goal in a couple of steps 
using provided taxonomy. You can take the help of taxonomy below to create new plan.

Taxonomy can be understood as follows: node [parent] is [relation] by [child], where parent and 
child are name of node in taxonomy and relation is the connection name. You are given an agent 
that can "update" the taxonomy relations and "craft" new recipe.

Here is are known connection in taxonomy.

taxonomy relations:
industrial asset health is root node
industrial asset health is analyzed by component health
industrial asset health is analyzed by historical record
industrial asset health is analyzed by asset profile
component health is impacted by mechanical issue
component health is impacted by electrical issue
component health is impacted by thermal health issue
component health is impacted by chemical health issue
mechanical issue is measured by on-demand inspection
mechanical issue is measured by continuous sensors
mechanical issue is measured by periodic chemical sampling

Goal: craft asset health recipe.

# Think: My target is asset health recipe. From the list of taxonomy relations, the root node is asset health and
it matched with my target. I will use this taxonomy relations to devise a plan. The taxonomy relations are
described as follow: Industrial asset health is analyzed by component health, historical record and asset profile. 
The component health is impacted by mechanical issue, electrical issue, thermal health issue and chemical 
health issue. The mechanical issue is measured by on-demand inspection, continuous sensors and periodic 
chemical sampling. I should first update taxonomy relations and then prepare the plan.
   
Step 1: update industrial asset health
Step 2: update component health
Step 3: update mechanical issue
# Think: Now that I have updated the taxonomy relations, I can craft the asset health recipe using updated taxonomy.
Step 4: craft asset health recipe using updated taxonomy relations 
# Think: To succeed, I need to perform all these steps, one after the other. So I need to use the "AND" operator.
Execution Order: (Step 1 AND Step 2 AND Step 3 AND Step 4)  Goal completed!

Goal: update industrial asset health.

# Think: My target is industrial asset health. From the list of taxonomy relations, industrial asset health is root 
node and it is connected to component health, historical record and asset profile. My current ingredients are: 
[component health, historical record,  asset profile]. To successfully accomplish the goal, I should search 
first get all the ingredients and then use the crafting command.

Step 1: is industrial asset health analyzed by component health?
Step 2: is industrial asset health analyzed by historical record?
Step 3: is industrial asset health analyzed by asset profile?
# Think: Now that I have verified the existing taxonomy relations, I can search for the new relations 
by repeating Step 3 one more time. I will not select any node from existing taxonomy relations. 
Step 4: display updated taxonomy relations. 
# Think: To succeed, I need to perform all these steps, one after the other. So I need to use the "AND" operator.
Execution Order: (Step 1 AND Step 2 AND Step 3 AND Step 4) Goal completed!

Here is a different goal with different taxonomy relations. Your task is to come up with a short plan to help 
me accomplish my goal in a couple of steps using provided taxonomy. You can take the help of taxonomy below to 
create new plan. Keep in mind that:
- It is okay to update taxonomy relations than the original.
- Be very careful with the adding of duplicate node.
- You cannot use a partial taxonomy relations.
- Also, you can use ONLY 1 crafting command in your plan.

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
        stop_sequences=["Goal completed!"],
        return_options=ReturnOptions(input_text=False, input_tokens=True),
        moderations=ModerationsOptions(
            # Threshold is set to very low level to flag everything (testing purposes)
            # or set to True to enable HAP with default settings
            hap=HAPOptions(input=True, output=False, threshold=0.01)
        ),
    ),
)

text = ["update industrial asset health"]
for sen in text:
    result = llm.generate(
        prompts=[
            f"System Prompt: {SystemPrompt}  \n\n Goal: {sen}."
        ]
    )
    print (result.generations[0][0].text)