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

Taxonomy can be understood as follows: [parent] is [relation] by [child], where parent and 
child are nodes in taxonomy and relation is the connection between parent and child nodes. 
Taxonomy is non-cyclic, it means same node cannot be connected to itself. You are given an agent
that can produce plan using given taxonomy.

Here is asset health taxonomy.

asset health taxonomy:
asset health is root node
asset health is analyzed by component
asset health is analyzed by historical record
asset health is analyzed by asset profile
component is impacted by mechanical issue
component is impacted by electrical issue
component is impacted by thermal health issue
component is impacted by chemical health issue
mechanical issue is measured by on-demand inspection
mechanical issue is measured by continuous sensors
mechanical issue is measured by periodic chemical sampling
electrical issue is measured by insulin
historical record is source by asset maintenance
historical record is source by failure
historical record is source by repair history
asset profile is recorded by age
asset profile is recorded by operating hours
asset profile is recorded by idle hours

Goal: calculate asset health using component.

# Think: My target is component. I will use asset health taxonomy to devise plan for component.
From the asset health taxonomy, there are four options that starts with node component. The component is impacted by 
mechanical issue, electrical issue, thermal health issue and chemical health issue. Subsequently, mechanical issue is 
measured by on-demand inspection, continuous sensors and periodic chemical sampling and so on. I should prepare the plan 
using given taxonomy. I will travese the taxonomy in top-down fashion starting from component node. 
   
Step 1: Let us focus on the component based asset health. We are 
    interested in the factors coming from the 1. Mechanical, 2. Electrical, 3. 
    Thermal, and 4. Chemical quality issues for the asset health. Can you give me detailed 
    guidelines for identifying factors impacting overall asset health?

# Think: Now, I will traverse taxonomy for each child node connected to component node. From the asset health taxonomy, 
component is impacted by mechanical issue and thus we will identify factors associated with mechanical issue. 
    
Step 2: Let us focus on mechanical issue of component. We are 
    interested in the factors coming from the 1. on-demand inspection, 2. continuous sensors, 
    and 3. periodic chemical sampling mechanic for the asset health. Can you give me detailed 
    guidelines for identifying factors impacting overall asset health?

# Think: Now that I have traversed mechanical issue, I should repeat Step 2 for any other node that is child of component. From the asset health taxonomy,
mechanical issue and electrical issue are only two children of component, and we already visited mechanical issue so we will ignore it.
We will repeat Step 2 for electrical issue only. I know that I cannot use a partial taxonomy relations. I will add one more step for each unvisited node.

Step 3: Collect all the discovered factors so far.  

Step 4: Let us assume that I am only interested in evaluating the asset health 
    based on the currently available data and discovered {factors}. I need your help to identify 
    the factors that indicate the deterioration of the {asset_class}'s component quality, 
    and those factors should be able to be monitored and quantified in the future  
    data.

# Think: We now try to be more focused on given asset class and asset description. 
Step 5: Great, this answer is an excellent general guideline. Let's focus on a 
    specific asset type: {asset_class} to analyze the asset component health on the 
    deterioration of the {asset_class}'s health score.  Here is a detailed explanation:{asset_description}. 
    Do not include any specific company contact information.

Step 6: We need to be very specific for asset type {asset_class}. Here is a detailed 
    explanation: {asset_description}. Let us focus on the factor, Key Performance Indicators (KPIs) 
    and reports. Please list all of them with the factors, KPIs and report, 
    explanation, and impact on asset health. 

# Think: To succeed, I need to perform all these steps, one after the other. So I need to use the "AND" operator.
Execution Order: (Step 1 AND Step 2 AND Step 3 AND Step 4 AND Step 5 AND Step 6) Goal completed!

Here is a different goal with different taxonomy relations. Your task is to come up with a short plan to help 
me accomplish my goal in a couple of steps using provided asset health taxonomy. You can take the help of asset health taxonomy below to 
create new plan. Keep in mind that:
- It is okay to update asset health taxonomy than the original.
- Be very careful with the adding of duplicate node in asset health taxonomy.
- You cannot use a partial asset health taxonomy.
- You cannot repeat same Step in plan.

"""
# Think: We now try to be very specific.

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v01-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
]

Recipes = ["calculate asset health using asset profile",
           "calculate asset health using historical record",
           "calculate asset health using component"]

for sen in Recipes:
    for llm in LLMsets:
        llm_mdl = LangChainInterface(
            model=llm,
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

        result = llm_mdl.generate(
            prompts=[
                f"System Prompt: {SystemPrompt}  \n\n Goal: {sen}."
            ]
        )
        print (f"Start----model {llm}, Recipe {sen}")
        print (result.generations[0][0].text.rstrip())
        print ('End -------------------------------')