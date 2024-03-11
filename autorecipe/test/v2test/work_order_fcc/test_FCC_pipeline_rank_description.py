from autorecipe.genai.GenAIChat import GenAIChatClient, extract_code
from autorecipe.genai.GenAIInstruct import GenAIInstructClient
import pandas as pd
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
the work order and entered the task description. User has provided asset short description, 
location and task description.

Asset Description: {assetdescription}

Asset Location: {assetlocation}

Failure Code: Here is a list of 23 failure codes, its full form and short descriptions. 

1. BUR, Burned, Consumed/damaged/deformed because of overheating.
2. CRK, Cracked, Damaged and showing lines on the surface from having split without coming apart.
3. DFM, Deformed, Not having the normal or natural shape or form.
4. ELK, External leak, Accidentally lose or admit contents. Source of leak is visible.
5. FRC, Fractured, Fractured or damaged and no longer in one piece.
6. FTC, Failure to close, equipment does not close on demand.
7. FTO, Failure to open, equipment does not open on demand.
8. FTS, Failure to start/function on demand, the equipment does not start/function on demand.
9. INL, Internal leak, Accidentally lose or admit contents. Source of leak is not visible.,
10. LFS, Lubrication/Fluid Sampling, Unsatisfactory fluid/lubrication sample.
11. MBU, Material build-up, Accumulation of  foreign/external material.
12. NOI, Noise, A sound that is loud and causes disturbance.
13. PDE, Parameter deviation, A parameter is not controlled as set.
14. PLU, Plugged, A constraint in flow conductor.
15. PTF, Power/signal failure, Power or signal loss in electrical system.
16. RUS, Rust, Build up of corrosion products.
17. SAL, Spurious Alarm, which is described as Unexpected/false alarm.
18. SLP, Slippage, Decrease of transmitted power in a mechanical system caused by slipping.
19. STP, Does not stop on demand, equipment does not stop on demand.
20. STU, Stuck, equipment does not move on demand.
21. TEX, Loss of Explosion Protection (EX) Integrity, Loss of Explosion Protection (EX) Integrity.
22. TRD, Thickness reduction, Reduction in thickness measurement / loss of material.
23. VIB, Fix Vibration, Vibration is higher that the established limit.

User will provide a work order description and your task is to provide ordering of these 23 failure codes.
The work order description is the description of the task that was conducted when work order is completed. The
work order description may captures failure modes that are associated with asset and its location. Your task
is to provide a ranking of the most important failure codes that can be associated with work order description,
asset description and its location.

"""

# most important --> relevant

workordertemplate = """Work order description: {taskdescription}"""

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

import ray
ray.init()

#prompt = SystemPrompt.format(assetdescription=assetdescription, assetlocation=assetlocation, taskdescription=taskdescription)
#print (prompt)

LLMsets = ['meta-llama/llama-2-70b-chat',
        'ibm/granite-13b-chat-v2',
        'ibm/granite-13b-labrador-rc',
        'ibm-mistralai/mixtral-8x7b-instruct-v01-q',
        #'ibm/granite-13b-instruct-v2',
        ]

for mdl in LLMsets:

    params = {
        "decoding_method": DecodingMethod.GREEDY,
        "min_new_tokens": 100,
        "max_new_tokens": 2000,  # 1500,
        "stop_sequences": ["(TOKENSTOP)"],
        "return_options": TextGenerationReturnOptions(input_text=False, input_tokens=True),
        "moderations": ModerationParameters(
            hap=ModerationHAP(input=True, output=False, threshold=0.01)
        ),
    }

    experiment_name = "MyExperiment_" + str(uuid.uuid4())
    experiment_id = mlflow.create_experiment(experiment_name)

    assetdescription = "NOV CYBERBASE DRILLING CONTROL"
    assetlocation = "DRILLING CONTROL"
    taskdescription = """Replace push button on Cyberspace C chair. Replace sticky push button on Cyber chair C."""

    df = pd.read_csv('./sample_data.csv')

    assetdescriptionLst = list(df['assetdescription'])
    locationdescriptionLst = list(df['locationdescription'])
    taskdescriptionLst = list(df['taskdescription'])
    problemcodeLst = list(df['problemcode'])

    @ray.remote
    def run_one(assetdescription, assetlocation, taskdescription):
        """
        pass the topic
        """
        Sprompt = SystemPrompt.format(assetdescription=assetdescription,
                                    assetlocation=assetlocation,
                                    taskdescription=taskdescription)

        qpromt = workordertemplate.format(taskdescription=taskdescription)

        tmpClient = GenAIChatClient(name='Testing',
                                        description='I am testing classifier',
                                        skill='Generic',
                                        model=mdl,
                                        params=params,
                                        credentials=creds,
                                        stateful=False,
                                        system_message=Sprompt)

        answer = tmpClient.create(
            context=None,
            messages=[{"content": qpromt, "role": "user"}],
            experiment_id=experiment_id,
        )
        #print (qpromt)
        answer = answer.strip()
        #print (answer)

        qpromt = """Prepare a human-readable one paragraph summary of given document in a 
        well-structured paragraph, eliminating any special characters such as new 
        lines and tabs. Focus on capturing the work order task description,
        asset description, summary of failure modes associated with work order description, 
        and important failure codes discussion. Ensure the summary provides a coherent narrative.
        Summary will be used to select the the best possible failure code that can be 
        assigned to the given work order.
        
        document: {answer}

        """

        json_res = tmpClient.create(
            context=None,
            messages=[{"content": qpromt.format(answer=answer), "role": "user"}],
            experiment_id=experiment_id,
        )
        #print (json_res.strip())
        return [answer, json_res.strip()]

    parallel_returns = []
    refs = []
    for i in range(len(assetdescriptionLst)):
    #for i in range(8):
        #'''
        refs.append(run_one.remote(assetdescriptionLst[i],
                                locationdescriptionLst[i],
                                taskdescriptionLst[i]))
        '''
        ans1 = run_one(assetdescriptionLst[i],
                    locationdescriptionLst[i],
                    taskdescriptionLst[i])
        print (problemcodeLst[i])
        parallel_returns.append(ans1)
        '''

    parallel_returns = ray.get(refs)
    print (parallel_returns)
    #continue

    res = pd.DataFrame(parallel_returns)
    finalans = pd.concat([df,res], axis=1)
    finalans.to_csv('Generated_result_'+ mdl.split('/')[1].replace('-','_') + '.csv',index=False)
    #print (finalans)

    '''
    for i in range(len(assetdescriptionLst)):
        print ('Start -------------------------------------')
        print (f'Ground Truth: {problemcodeLst[i]}')
        print (parallel_returns[i])
    '''