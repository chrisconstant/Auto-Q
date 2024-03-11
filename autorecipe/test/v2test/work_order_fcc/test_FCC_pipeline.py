from autorecipe.genai.GenAIChat import GenAIChatClient
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationReturnOptions,
)
import uuid
import mlflow

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
creds = {
    "api_key": "pak-KnqohFjgmJKou_eirLnIJMtbxdFzUYylnzScnLxVhY0",
    "api_endpoint": "https://bam-api.res.ibm.com",
}

SystemPrompt = """I am working on failure code classification problem. The technician has completed 
the work order and entered the task description. user has provided asset short description, 
location and task description.

Asset Description: {assetdescription}

Asset Location: {assetlocation}

Task description: {taskdescription}

"""

PromptSets = [
    """You need to generate the better task description that is redable.""", 
    """The task description is the description of the task that was conducted when work order is completed.""",
    """What are the potential failure modes discussed in work order task description that are associated with asset and its location?""",
    """Please provides a list of the important failure modes discussed in task description?""",]

FCodes = [("BUR","Burned, which is described as Consumed/damaged/deformed because of overheating."),
          ("CRK","Cracked, which is described as Damaged and showing lines on the surface from having split without coming apart."),
          ("DFM","Deformed, which is described as Not having the normal or natural shape or form."),
          ("ELK","External leak, which is described as Accidentally lose or admit contents. Source of leak is visible."),
          ("FRC","Fractured, which is described as Fractured or damaged and no longer in one piece."),
          ("FTC","Failure to close, which is described as The equipment does not close on demand."),
          ("FTO", "Failure to open, which is described as The equipment does not open on demand."),
          ("FTS","Failure to start/function on demand, which is described as The equipment does not start/function on demand."),
          ("INL","Internal leak, which is described as Accidentally lose or admit contents.  Source of leak is not visible."),
          ("LFS","Lubrication/Fluid Sampling, which is described as Unsatisfactory fluid/lubrication sample."),
          ("MBU", "Material build-up, which is described as Accumulation of  foreign/external material."),
          ("NOI", "Noise, which is described as A sound that is loud and causes disturbance."),
          ("PDE","Parameter deviation, which is described as A parameter is not controlled as set."),
          ("PLU","Plugged, which is described as A constraint in flow conductor."),
          ("PTF","Power/signal failure, which is described as Power or signal loss in electrical system."),
          ("RUS","Rust, which is described as Build up of corrosion products."),
          ("SAL","Spurious Alarm, which is described as Unexpected/false alarm."),
          ("SLP", "Slippage, which is described as Decrease of transmitted power in a mechanical system caused by slipping."),
          ("STP", "Does not stop on demand, which is described as The equipment does not stop on demand."),
          ("STU","Stuck, which is described as Unable to move."),
          ("TEX","Loss of Explosion Protection (EX) Integrity, which is described as Loss of Explosion Protection (EX) Integrity."),
          ("TRD", "Thickness reduction, which is described as Reduction in thickness measurement / loss of material."),
          ("VIB","Vibration, which is described as Vibration is higher that the established limit.")
] 

LLMsets = ['ibm/granite-13b-chat-v2',
        'meta-llama/llama-2-70b-chat',
        'ibm-mistralai/mixtral-8x7b-instruct-v01-q']

params = {
    "decoding_method": DecodingMethod.GREEDY,
    "min_new_tokens": 100,
    "max_new_tokens": 500,  # 1500,
    "stop_sequences": ["(TOKENSTOP)"],
    "return_options": TextGenerationReturnOptions(input_text=False, input_tokens=True),
    "moderations": ModerationParameters(
        hap=ModerationHAP(input=True, output=False, threshold=0.01)
    ),
}

experiment_name = "MyExperiment_" + str(uuid.uuid4())
experiment_id = mlflow.create_experiment(experiment_name)

import pandas as pd
df = pd.read_csv('./sample_data.csv')
assetdescriptionLst = list(df['assetdescription'])
locationdescriptionLst = list(df['locationdescription'])
taskdescriptionLst = list(df['taskdescription'])

import ray
ray.init()

@ray.remote
def run_one(assetdescription, assetlocation, taskdescription, model_id=0):
    """
    pass the topic
    """
    ansSet = []
    Sprompt = SystemPrompt.format(assetdescription=assetdescription,
                                  assetlocation=assetlocation,
                                  taskdescription=taskdescription)

    tmpClient = GenAIChatClient(name='Testing',
                                    description='I am testing classifier',
                                    skill='Generic',
                                    model=LLMsets[model_id],
                                    params=params,
                                    credentials=creds,
                                    system_message=Sprompt,
                                    stateful=True)

    for _, qpromt in enumerate(PromptSets):
        answer = tmpClient.create(
            context=None,
            messages=[{"content": qpromt, "role": "user"}],
            experiment_id=experiment_id,
        )
        ansSet.append(answer.strip())
        #print (answer)

    fcode_template = """
    I have following failure codes: 

    Failure Code: {fc}
    Description: {fdesc}.

    Does these failure code is relevant to the work order task description?

    """
    # Now we will move into looping all the failure mode one after another
    for fcode in FCodes:
        #print ('Start...-----------------------------------------------------------------------------')
        fccode = fcode_template.format(fc=fcode[0], fdesc=fcode[1])
        answer = tmpClient.create(
            context=None,
            messages=[{"content": fccode, "role": "user"}],
            experiment_id=experiment_id,
        )
        answer = answer.strip()
        ansSet.append(answer)
    return ansSet

for model_id in range(len(LLMsets)):
    refs = []
    for i in range(len(assetdescriptionLst)):
        refs.append(run_one.remote(assetdescriptionLst[i],
                                locationdescriptionLst[i],
                                taskdescriptionLst[i],
                                model_id))
    parallel_returns = ray.get(refs)

    res = pd.DataFrame(parallel_returns)
    finalans = pd.concat([df,res], axis=1)

    model_initial = LLMsets[model_id].split("/")[1].split("-")[0]
    finalans.to_csv(f'step1_generated_result_{model_initial}.csv',index=False)
    # print (finalans)