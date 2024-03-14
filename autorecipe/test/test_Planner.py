import os

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from genai.credentials import Credentials
from genai.extensions.langchain import LangChainInterface
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.schemas import ChatOptions, GenerateParams, ReturnOptions
from genai.schemas.generate_params import HAPOptions, ModerationsOptions

"""

asset sustainability taxonomy:
asset sustainability is root node
asset sustainability is analyzed by greenhouse impact
asset sustainability is analyzed by resources impact
asset sustainability is contributed by environmental impact
environmental impact is influenced by air pollution
environmental impact is influenced by water pollution
environmental impact is influenced by solid waste
environmental impact is influenced by noise level
environmental impact is influenced by ecological consequence
greenhouse impact is measured by co2-equivalent emissions
resources impact is depending on non-renewable resources

"""

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
api_key = "pak-KnqohFjgmJKou_eirLnIJMtbxdFzUYylnzScnLxVhY0"
api_endpoint = "https://bam-api.res.ibm.com"


CodeGenerationPrompt = """
Your task is to come up with a short plan to help me accomplish my goal in a couple of steps 
using provided taxonomy. You can take the help of taxonomy below to create new plan.

Taxonomy can be understood as follows: [parent] is [relation] by [child], where parent and 
child are nodes in taxonomy and relation is the connection between parent and child nodes. 
Taxonomy is non-cyclic, it means same node cannot be connected to itself. You are given an agent
that can produce plan by traversing given taxonomy recursively.

Here is asset health taxonomy.

asset health taxonomy:
asset health is root node
asset health is analyzed by component quality
asset health is analyzed by historical record
asset health is analyzed by asset profile
component quality is impacted by mechanical issue
component quality is impacted by electrical issue
component quality is impacted by thermal health issue
component quality is impacted by chemical health issue
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

Here is asset sustainability taxonomy:
asset sustainability is root node
asset sustainability is analyzed by greenhouse impact
asset sustainability is analyzed by resources impact
asset sustainability is contributed by environmental impact
environmental impact is influenced by air pollution
environmental impact is influenced by water pollution
environmental impact is influenced by solid waste
environmental impact is influenced by noise level
environmental impact is influenced by ecological consequence
greenhouse impact is measured by co2-equivalent emissions
resources impact is depending on non-renewable resources


Python Code:
```python
def callme(node='asset sustainability'):
    taxonomy = {
        'asset sustainability': ['greenhouse impact','resources impact','environmental impact'],
        'environmental impact': ['air pollution','water pollution','solid waste','noise level','ecological consequence'],
        'greenhouse impact': ['co2-equivalent emissions'],
        'resources impact': ['non-renewable resources'],
        'asset health': ['component', 'historical record', 'asset profile'],
        'component': ['mechanical issue', 'electrical issue', 'thermal health issue', 'chemical health issue'],
        'mechanical issue': ['on-demand inspection', 'continuous sensors', 'periodic chemical sampling'],
        'electrical issue': ['insulin'],
        'historical record': ['asset maintenance', 'failure', 'repair history'],
        'asset profile': ['age', 'operating hours', 'idle hours'],
        'on-demand inspection': [],
        'continuous sensors': [],
        'periodic chemical sampling': [],
        'insulin': [],
        'asset maintenance': [],
        'failure': [],
        'repair history': [],
        'age': [],
        'operating hours': [],
        'idle hours': [],
    }

    def _check_node(node, taxonomy):
        if 'sensor' in node:
            return 'yes it is there'
        if node not in taxonomy:
            return 'skip'
        for child in taxonomy[node]:
            result = _check_node(child, taxonomy)
            if result == 'yes it is there':
                return 'yes it is there'
        return 'skip'

    return _check_node(node, taxonomy)
```

"""

stepcommand = """Generate a Python code that takes component as an input node from taxonomy and returns "yes it is there" if any
child node including the input node has a word sensor into their name otherwise it returns skip. Please handle KeyError, especially 
prior to dictinory key. Only generate Python code in PEP standard, Do not generate an explanation. Generate a single python function as entry
point : callme(node='component')."""

SystemPrompt = """
Your task is to come up with a short plan to help me accomplish my goal in a couple of steps 
using provided taxonomy. You can take the help of taxonomy below to create new plan.

Taxonomy can be understood as follows: [parent] is [relation] by [child], where parent and 
child are nodes in taxonomy and relation is the connection between parent and child nodes. 
Taxonomy is non-cyclic, it means same node cannot be connected to itself. You are given an agent
that can produce plan by traversing given taxonomy recursively.

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

Here is asset sustainability taxonomy:
asset sustainability is root node
asset sustainability is analyzed by greenhouse impact
asset sustainability is analyzed by resources impact
asset sustainability is contributed by environmental impact
environmental impact is influenced by air pollution
environmental impact is influenced by water pollution
environmental impact is influenced by solid waste
environmental impact is influenced by noise level
environmental impact is influenced by ecological consequence
greenhouse impact is measured by co2-equivalent emissions
resources impact is depending on non-renewable resources

Goal: calculate asset health using component.

# Think: My target is component. I will use asset health taxonomy to devise plan for component.
From the asset health taxonomy, there are four options that starts with node component. The component is 
impacted by mechanical issue, electrical issue, thermal health issue and chemical health issue. 
Subsequently, mechanical issue is measured by on-demand inspection, continuous sensors and periodic chemical 
sampling and so on. I should prepare the plan using given taxonomy. I will travese the taxonomy in 
top-down fashion starting from component node. 
   
Step 1: Let us focus on the component based asset health. We are 
    interested in the factors coming from the 1. Mechanical, 2. Electrical, 3. 
    Thermal, and 4. Chemical quality issues for the asset health. Can you give me detailed 
    guidelines for identifying factors impacting overall asset health?

# Think: Now, I will traverse taxonomy for each child node connected to component node and so on. From the asset health taxonomy, 
component is impacted by mechanical issue and thus we will visit mechanical issue node to identify factors associated with mechanical issue.
I know that I cannot use a partial recipe.   

Step 2: Let us focus on mechanical issue of component. We are 
    interested in the factors coming from the 1. on-demand inspection, 2. continuous sensors, 
    and 3. periodic chemical sampling mechanic for the asset health. Can you give me detailed 
    guidelines for identifying factors impacting overall asset health?

# Think: Now that I have visited mechanical issue node, I will focus on other child node connected to 
component node. From the asset health taxonomy, component is also impacted by electrical issue, thermal health issue and 
chemical health issue. I know that I cannot use a partial recipe. So my goal is not satisfied, I need to visit more 
nodes by repeating Step 2 three more times.
 
Step 3: Let us focus on electrical issue of component. We are 
    interested in the factors coming from the 1. insulin for the asset health. Can you give me detailed 
    guidelines for identifying factors impacting overall asset health?
    
Step 4: Let us focus on thermal health issue of component. We are 
    interested in the factors coming from the thermal health issue of component for the asset health. Can you give me detailed 
    guidelines for identifying factors impacting overall asset health?

Step 5: Let us focus on chemical health issue of component. We are 
    interested in the factors coming from the chemical health issue of component for the asset health. Can you give me detailed 
    guidelines for identifying factors impacting overall asset health?
        
# Think: We have visited all four nodes: mechanical issue, electrical issue, thermal health issue and chemical health issue. 
Now that I have traversed all the child and subchild node of component node. I should repeat Step 2 for any other node that is not visited so far. 
I know that I know that I cannot use a partial recipe.

Step 6: Collect all the discovered factors so far.  

# Think: We now shift our focus on discovered factors, asset class and asset description. 

Step 7: Let us assume that I am only interested in evaluating the asset health 
    based on the currently available data and discovered {factors}. I need your help to identify 
    the factors that indicate the deterioration of the {asset_class}'s component quality, 
    and those factors should be able to be monitored and quantified in the future  
    data.

Step 8: Great, this answer is an excellent general guideline. Let's focus on a 
    specific asset type: {asset_class} to analyze the component health on the 
    deterioration of the {asset_class}'s health score.  Here is a detailed explanation:{asset_description}. 
    Do not include any specific company contact information.

Step 9: We need to be very specific for asset type {asset_class}. Here is a detailed 
    explanation: {asset_description}. Let us focus on the factor, Key Performance Indicators (KPIs) 
    and reports. Please list all of them with the factors, KPIs and report, 
    explanation, and impact on asset health. 

Step 10: Generate a Python code that takes component as an input node from taxonomy and returns "yes it is there" if any
child node of the input node has a word sensor into their name otherwise it returns skip. Code will travese the 
taxonomy in top-down fashion starting from input node. Please handle KeyError, especially 
prior to dictinory key. Use dictionary to store the parent and child relationship of taxonomy.  
Leaf node should also present in dictionary with empty child. Only generate Python code in PEP standard, 
Do not generate an explanation. Generate a single python function as entry
point callme(node='component'). Any explanation please put them into python code.
    
# Think: To succeed, I need to perform all these steps, one after the other. So I need to use the "AND" operator.
Execution Order: (Step 1 AND Step 2 AND Step 3 AND Step 4 AND Step 5 AND Step 6 AND Step 7 AND Step 8 AND Step 9 AND Step 10) Goal completed!

Here is a different goal with different taxonomy relations. Your task is to come up with a short plan to help 
me accomplish my goal in a couple of steps using provided asset health taxonomy. You can take the help of asset health taxonomy below to 
create new plan. Keep in mind that:
- It is okay to update asset health taxonomy than the original.
- Be very careful with the adding of duplicate node in asset health taxonomy.
- You cannot use a partial asset health taxonomy.
- You cannot repeat same Step in plan.
- Do not traverse newly created node in taxonomy.

"""
# Think: We now try to be very specific.

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v0-1-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
]

Recipes = ["calculate asset sustainability using environmental impact",
    "calculate asset health using asset profile",
           "calculate asset health using historical record",
           "calculate asset health using asset component"]

Recipes = [Recipes[0]]

def generate_code_execute(code_prompt):
    llm_mdl = LangChainInterface(
        model=LLMsets[0],
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
            f"System Prompt: {CodeGenerationPrompt}  \n\n Question: {code_prompt} \n\n Answer:"
        ]
    )
    answer = result.generations[0][0].text.rstrip()
    print (answer)
    from autorecipe.genai.GenAIChat import extract_code 
    answer = extract_code(answer)

    local_vars = {}
    exec(answer[0][1],globals(), local_vars)
    callme = local_vars['callme']
    result = callme(node='asset sustainability')
    print (result)
    return result

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
        answer = result.generations[0][0].text.rstrip()

        import re
        # Remove lines starting with "# Think:"
        cleaned_text = re.sub(r'# Think:.*\n', '', answer)
        cleaned_text = re.sub(r'Execution Order:.*\n', '', cleaned_text)

        # Use regular expression to find all steps and their content
        pattern = re.compile(r'(Step \d+:.*?)(?=Step \d+:|$)', re.DOTALL)
        matches = pattern.findall(cleaned_text)

        # Resulting list of steps and their content
        steps_list = [step.strip() for step in matches]

        # Display the list of steps
        fianl_steps = []
        for step in steps_list:
            if "Generate a Python code" in step:
                # ans1 = generate_code_execute(step)
                # if type(ans1) == bool and ans1:
                #     fianl_steps.append(step.split(' ')[0] + ' What kinds of sensor could be used to check such quality deterioration or measure such KPIs?')
                # elif type(ans1) == str and "yes" in ans1.lower():
                #     fianl_steps.append(step.split(' ')[0] + ' What kinds of sensor could be used to check such quality deterioration or measure such KPIs?')
                # else:
                    pass
            else:
                fianl_steps.append(step)
        for step in fianl_steps:
            print ('S---')
            print (step)
            print ('---E')

        print ('End -------------------------------')

"""
historical record
# Think: We now proceed to final step of report generation 
Step 10: We would like to export the above into a markdown file.  Please help us to 
    generate that.  If needed, you can use the markdown table.  
    At the beginning of the markdown file,  please explain how the historical 
    records help assess the {asset_class}'s asset health assessment.

asset_age
Please help me summarize the information above into a markdown format 
    output as guidelines for analyzing the asset aging (including the operation hours)
    If needed, you can use the table format inside of the markdown file.  
    The output has two  parts:
    1. It is the beginning part of the document. Please briefly introduce 
    the Hydraulic Press, its usage in various applications and also 
    include the introduction of its asset;
    2. The body part include the impact of the asset aging factors to the 
    deterioration of the {asset_class} asset health.        

Sustainability
I would like to export the above into a markdown file.  Please help me to 
generate that.  If needed, you can use the markdown table.  
At the beginning of the markdown file,  please explain how those factors 
help assess the {asset_class}'s sustainability. The includes:
Part 1: the factors that indicate the factors impacting the sustainability
of an asset such as {asset_class}. 
Part 2: highlights the sensors deployed or being able to measure KPIs for 
sustainability.  Please output as a list, each list containing a) 
the sustainability being monitored, b) the sensor used, and c) the reason 
for using such sensor.

Please help me export a markdown output as guidelines for analyzing the quality of
    asset component health or quality to overall asset health, the output has three parts:
    Part 1:  It is the beginning part of the document, please briefly introduce the {asset_class}, its usage in the 
    various application and also include the introduction of its component
    Part 2: the factors/KPIs that indicate or impact the asset health such as the deterioration of the {asset_class}'s 
    asset health; 
    Part 3 (optional): highlights the sensors deployed or being able to use measure such components quality 
    or its quality deterioration.  Please output as a list, each list contains a) the quality problem being 
    monitored; b) the sensor used, and c) the reason of using such sensor. 

What kinds of sensor could be used to check such quality deterioration or measure such KPIs?

"""


