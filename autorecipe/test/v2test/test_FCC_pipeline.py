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
    "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
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

FCodes = [("BUR","Fix Burned, which is described as Consumed/damaged/deformed because of overheating."),
          ("CRK","Fix Cracked, which is described as Damaged and showing lines on the surface from having split without coming apart."),
          ("DFM","Fix Deformed, which is described as Not having the normal or natural shape or form."),
          ("ELK","Fix External leak, which is described as Accidentally lose or admit contents. Source of leak is visible."),
          ("FRC","Fix Fractured, which is described as Fractured or damaged and no longer in one piece."),
          ("FTC","Fix Failure to close, which is described as The equipment does not close on demand."),
          ("FTO", "Fix Failure to open, which is described as The equipment does not open on demand."),
          ("FTS","Fix Failure to start/function on demand, which is described as The equipment does not start/function on demand."),
          ("INL","Fix Internal leak, which is described as Accidentally lose or admit contents.  Source of leak is not visible."),
          ("LFS","Fix Lubrication/Fluid Sampling, which is described as Unsatisfactory fluid/lubrication sample."),
          ("MBU", "Fix Material build-up, which is described as Accumulation of  foreign/external material."),
          ("NOI", "Fix Noise, which is described as A sound that is loud and causes disturbance."),
          ("PDE","Fix Parameter deviation, which is described as A parameter is not controlled as set."),
          ("PLU","Fix Plugged, which is described as A constraint in flow conductor."),
          ("PTF","Fix Power/signal failure, which is described as Power or signal loss in electrical system."),
          ("RUS","Fix Rust, which is described as Build up of corrosion products."),
          ("SAL","Fix Spurious Alarm, which is described as Unexpected/false alarm."),
          ("SLP", "Fix Slippage, which is described as Decrease of transmitted power in a mechanical system caused by slipping."),
          ("STP", "Fix Does not stop on demand, which is described as The equipment does not stop on demand."),
          ("STU","Fix Stuck, which is described as Unable to move."),
          ("TEX","Fix Loss of Explosion Protection (EX) Integrity, which is described as Loss of Explosion Protection (EX) Integrity."),
          ("TRD", "Fix Thickness reduction, which is described as Reduction in thickness measurement / loss of material."),
          ("VIB","Fix Vibration, which is described as Vibration is higher that the established limit.")
] 

LLMsets = ['ibm/granite-13b-chat-v2',
        'meta-llama/llama-2-70b-chat',
        'ibm/granite-13b-labrador-rc',
        'ibm-mistralai/mixtral-8x7b-instruct-v01-q',
        'ibm/granite-13b-instruct-v2']

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

assetdescription = "DECK CRANE 2, AHC KBC PORT FWD"
assetlocation = "361.DCR2"
taskdescription = """*FOLLLOW UP, MPI* 12M AHC KNUCKLE BOOM CRANE MARINE INSPECTION. Work with NDT inspectors 
to carry out MPI Inspection on repaired area inside Crane Pedestal. 2 X inboard corner doubler plates on 
stiffener at transition area from square to Pedestal cylinder. Areas were left without paint for to allow for 
further inspection. If no indications noticed then paint can be applied. If indication recorded notify TSL 
for further investigation. NDT report to be uploaded to edocs and attached."""

assetdescription = "MHWIRTH BC01 BRIDGE RACKING CRANE (BRC)"
assetlocation = "341.PRS1.110"
taskdescription = """BRC- Replace lift cylinder hold valve. Replace load hold valve due to leaking."""

assetdescription = "NOV CYBERBASE DRILLING CONTROL"
assetlocation = "DRILLING CONTROL"
taskdescription = """Replace push button on Cyberspace C chair. Replace sticky push button on Cyber chair C."""

import pandas as pd
df = pd.read_csv('./sample_data.csv')

assetdescriptionLst = list(df['assetdescription'])
locationdescriptionLst = list(df['locationdescription'])
taskdescriptionLst = list(df['taskdescription'])
problemcodeLst = list(df['problemcode'])


import ray
ray.init()

@ray.remote
def run_one(assetdescription, assetlocation, taskdescription):
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
                                    model=LLMsets[1],
                                    params=params,
                                    credentials=creds,
                                    system_message=Sprompt,
                                    stateful=True)

    for qindex, qpromt in enumerate(PromptSets):
        #print ('Start...-----------------------------------------------------------------------------')
        #print (qpromt)
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

refs = []
for i in range(len(assetdescriptionLst)):
#for i in range(8):
    refs.append(run_one.remote(assetdescriptionLst[i],
                               locationdescriptionLst[i],
                               taskdescriptionLst[i]))
parallel_returns = ray.get(refs)

res = pd.DataFrame(parallel_returns)
finalans = pd.concat([df,res], axis=1)

finalans.to_csv('Generated_result_1.csv',index=False)
print (finalans)

'''
for i in range(len(assetdescriptionLst)):
    print ('Start -------------------------------------')
    print (taskdescriptionLst[i])
    print (f'Ground Truth: {problemcodeLst[i]}')
    print (parallel_returns[i])
'''
