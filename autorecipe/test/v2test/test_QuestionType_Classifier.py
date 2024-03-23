from autorecipe.genai.GenAIInstruct import GenAIInstructClient
import pandas as pd
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationReturnOptions,
)

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
creds = {
    "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
    "api_endpoint": "https://bam-api.res.ibm.com",
}

SystemPrompt = """
You are a helpful, respectful, and honest assistant. You will be introduced to a 
failure code such as PTF and its description. The user will provide a workorder and you will analyze 
wether the workorder can be classified to PTF code or not using the provided annotation guideline. 
The description for failure code is given as follows.  

Failure Code: PTF
Failure Description: Power or signal loss or failure in electrical system. 
Source of leak is visible.

Workorder example that can be assigned to PTF:
1. Investigate encoder fault on MW DW D
2. Replace interface adapter/fiber ethernet TERM 1

"""

ClassifierPrompt = """
Assuming the user is seeking guidance about the failure code of the given work order. There is a possibility
that failure codes can not be associated with a work order.

Given the work-order, you will generate its short description. Then, you will compare the failure description 
with the generated description to determine wether the short description can be explained using Failure Code.
In addition, you can also compare the given work order with two workorder examples given in System prompt to 
decide the suitability of work order to give ELK failure code. Please generate explanation if you assing ELK 
failure code. If work order is not associated with ELK, then return NoCode.
 
"""

SystemPrompt12 = """
You are a helpful, respectful, and honest assistant. You will be introduced to several 
failure codes such as BUR, CRK, DFM, etc. The user will provide a 
question and you will recommed a most important failure code using the provided annotation guideline. The
definition for each failure code is given as follows. 

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

"""

ClassifierPrompt = """

Assuming the user is seeking guidance about the failure code of the given work order. There is a possibility
that more than one failure codes can be associated with a work order. 

Work Order: Mud bucket seals replaced due to worn parts. Please use (Internal thought).

(Internal thought): first we generate a more readable work order task description based on the information 
provided. 

Work Order Task Description: Replaced worn seals on mud bucket (MUD BUCKET) located at MAIN WELL, MUD SAVER 
BUCKET. The task was completed to ensure proper functioning of the mud bucket and prevent any potential issues
that may arise from worn parts. This description provides a clear and concise summary of the work that was done,
including the location of the asset and the specific task that was completed. It also includes the reason for 
the task, which can help provide context and justify the work that was done."

Next, Based on the information provided in the work order task description, the following are some potential 
failure modes details associated with the work order, asset and its location:

Failure Modes Detail:
1. Worn seals: The task description mentions that the seals on the mud bucket were replaced due to wear. This suggests that the asset may have experienced wear and tear over time, which could lead to leaks or other issues that could impact its performance.
2. Leaks: Leaks are a common failure mode associated with worn seals. If the seals are not properly maintained, they can allow mud or other fluids to leak out, which can lead to a range of problems, including environmental damage, safety hazards, and equipment malfunction.
3. Equipment malfunction: Worn seals or leaks can also lead to equipment malfunction, as the mud bucket may not be able to function properly if it is not properly sealed. This could lead to downtime, lost productivity, and additional maintenance costs.
4. Safety hazards: Leaks or other issues with the mud bucket can create safety hazards for personnel working in the area. For example, leaks can create slip and fall hazards, while malfunctioning equipment can pose a risk to operators or others nearby.
5. Environmental damage: Leaks or other issues with the mud bucket can also lead to environmental damage, as mud or other fluids can spill into the surrounding environment. This can have serious consequences, particularly in sensitive ecosystems or areas with strict environmental regulations.
These failure modes are associated with the asset (mud bucket) and its location (MAIN WELL, MUD SAVER BUCKET). 
The task description highlights the importance of addressing these failure modes by replacing the worn seals 
to prevent leaks, equipment malfunction, safety hazards, and environmental damage.

Next, I summarize some important failure modes detail discussed in the failure mode detail:
1. Worn seals
2. Leaks
3. Equipment malfunction
4. Safety hazards
5. Environmental damage

Finally, I find out the list of candidate problem codes mentioned in System Prompt. 

We found twenty three failure codes listed in system prompt: [BUR, CRK, DFM, ELK, FRC, FTC, FTO, FTS, INL, 
LFS, MBU, NOI, PDE, PLU, PTF, RUS, SAL, SLP, STP, STU, TEX, TRD, VIB] 

First, we process first failure code BUR (Burned). The task description mentions worn seals, leaks, 
equipment malfunction, safety hazards, and environmental damage, but it does not mention anything about 
overheating or burned parts. Therefore, the failure code "BUR" does not seem to be applicable to this 
particular work order.  

Next, we process second failure code CRK (Cracked). The task description mentions worn seals, leaks, 
equipment malfunction, safety hazards, and environmental damage, and cracked or damaged parts could 
contribute to any of these failure modes. For example, a cracked mud bucket could leak, causing 
environmental damage or safety hazards, or it could malfunction, leading to downtime and lost productivity. 
Therefore, the failure code "CRK" could be applied to this work order, 
as it addresses the issue of cracked or damaged parts that may be contributing to the observed failure modes.

Next, we process third failure code DFM (Deformed). The task description mentions worn seals, leaks, 
equipment malfunction, safety hazards, and environmental damage, and deformed parts could contribute to 
any of these failure modes. For example, a deformed mud bucket could leak, causing environmental damage 
or safety hazards, or it could malfunction, leading to downtime and lost productivity. In addition, the 
description of the failure code ""DFM"" specifically mentions that the asset may not have the normal or 
natural shape or form, which could be the case if the mud bucket is deformed due to wear or damage. Therefore, 
the failure code ""DFM"" could be applied to this work order, as it addresses the issue of deformed parts 
that may be contributing to the observed failure modes.

Overall, CRK, DFM, etc are candidate failure codes.

Answer: The final answer is [CRK, DFM]. (TOKENSTOP)

"""

ClassifierPrompt1 = """
Assuming the user is seeking guidance about the problem code of the given work order. There is a possibility
that more than one problem codes can be associated with a work order. Your answer should include 
all problem codes.

Work Order: Blue pod Can # 2 Power supply fault. Investigate report of P/S failure alarm reported by SSS. 
complete troubleshoot and replace/repair power supply. Return system to normal service. Generate at max 
top three problem codes that are associated with given work order. Please use (Internal thought).

(Internal thought): first we find out the list of candidate problem codes mentioned in System Prompt. 

We found twenty two failure codes listed in system prompt: [BUR, CRK, DFM, ELK, FRC, FTC, FTO, FTS, INL, 
LFS, MBU, NOI, PDE, PLU, PTF, RUS, SAL, SLP, STP, STU, TEX, TRD, VIB] 

Next, we generate short descriptive summary of given Work Order. The SSS has reported a P/S failure alarm 
for Blue Pod Can #2. Investigation reveals that the power supply is malfunctioning, with symptoms 
indicating a possible fracture or damage to the power supply components. The power supply is not 
functioning properly, and the system is not operating within normal parameters. 

Based on the work order description, a failure code of FRC (Fractured) or PTF (Power/signal failure) may 
be assigned to this issue.

Overall, FRC and PTF has positive sentiment.

Answer: The final answer is FRC or PTF. (TOKENSTOP)

"""

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v01-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
]

#Is this question for Subject Matter Expert?
#Is this question for Data Scientist?

instructions = [
    #'BOP#1  Blue pod Can # 2 Power supply fault. Investigate report of P/S failure alarm reported by SSS. complete troubleshoot and replace/repair power supply. Return system to normal service',
    #'BRC- Replace lift cylinder hold valve. Replace load hold valve due to leaking',
    'Replace interface adapter/fiber ethernet. Replace interface adapter/fiber ethernet TERM 1. Generate at max top three problem codes that are associated with given work order. ',
    #'AUX HR Pipe Pusher Prox Adjustment. AUX HR Upper Arm Guide Pipe Pusher Prox Adjustment',
    'Replace Inclination sensor crane 4 as Yoke. Horizon angle sensor 1 defect %W4.41.3. Generate at max top three problem codes that are associated with given work order. ',
    'Replace faulty encoder on MW DW Motor D. Investigate encoder fault on MW DW D. Trend encoder measurements on drive windows and compare. Replace encoder and check feedback on drive windows. Place DW D back in service and perform layer shift to put encoder D back in voting. Generate at max top three problem codes that are associated with given work order. ',
    'Replace Standpipe #2 Pressure transmitter on Main Well. Replace pressure transmitter STP #2 at Stand pipe Manifold Main Well. Generate at max top three problem codes that are associated with given work order. ',
]

instructions = [
    "Replace over center valve on the LGA jib tilt cylinder due to valve leaking.",
    "Replace leaking cylinder, PORT FWD",
    "Change out seal from gripper cylinder due to oil leakage",
    "replace 20 leaking air solenoid valves on lower fingerboard cabinets, due to leaking air",
    "Rebuild leaking cylinder on BOP#1 WHC",
    "replace seals on the valves",
    "ICN#1900357 - GASKET, 6-5/8IN: 3 ea To replaced worn rubbers that were observed to be leaking.",
    "Issue of parts for Mud Pump 4 repairs",
    "Change out test stump cylinder",
    "Replace Mechanical Seal",
    "Troubleshoot leak on choke line during riser run and pressure test",
    "Replace leaking slip joint packer air pressure regulator (R10) on diverter panel",
    "replace swab on #3 mud pump",
    "REPLACE LMRP GASKET RETAIN HYDRAULIC CYLINDER THAT IS LEAKING DURING SOAK TEST",
]

instructions = [
    "Horizon angle sensor 1 defect %W4.41.3.",
    "Replace pressure transmitter STP #2 at Stand pipe Manifold Main Well",
    "Alex Reyes- ET's have diagnosed several defective parts for SB HRN Cavotec remote. Parts are not in our inventory and will transfer from catalog. order QTY-2 of each.",
    "Replace Hydrostatic Transducer for PTM #2",
    "Replace Proxy Sensor of the MW Hydratong Mud Bucket Park Position",
    "Cable protection damage on crane 1 knuckle. install junctionbox on both sides of the knuckle and replace hose/cable temporary fix has been implemented.",
    "Replace batteries for BOP 2 Acoustic Control Unit with new ICN:1923761",
    "Replace Complete Transitter - 4-2- mA signal into CAN module checked by simulation",
]

params = {
    "decoding_method": DecodingMethod.GREEDY,
    "min_new_tokens": 200,
    "max_new_tokens": 2000,  # 1500,
    "stop_sequences": ["(TOKENSTOP)"],
    "return_options": TextGenerationReturnOptions(input_text=False, input_tokens=True),
    "moderations": ModerationParameters(
        hap=ModerationHAP(input=True, output=False, threshold=0.01)
    ),
}

import uuid
import mlflow
experiment_name = "MyExperiment_" + str(uuid.uuid4())
experiment_id = mlflow.create_experiment(experiment_name)

tmpClient = GenAIInstructClient(name='Testing',
                                description='I am testing classifier',
                                skill='Generic',
                                model=LLMsets[1],
                                params=params,
                                credentials=creds,
                                system_message=SystemPrompt,
                                question_message=ClassifierPrompt,
                                stream=False)

for qindex, qpromt in enumerate(instructions):
    print ('Start...-----------------------------------------------------------------------------')
    print (qpromt)
    #result = tmpClient.generate(
    #    prompts=[f"System Prompt: {SystemPrompt} \n\n {ClassifierPrompt} \n\n Question: {qpromt} Please use (Internal thought)."]
    #)
    answer = tmpClient.create(
        context=None,
        messages=qpromt,
        experiment_id=experiment_id,
    )
    print (answer)
    sindex = answer.rfind('Answer:')
    eindex = answer.rfind('(TOKENSTOP')
    print (answer[sindex+7:eindex])   # 7 = len('Answer:')
    print ('-----------------------------------------------------------------------------...End')
