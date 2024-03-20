from autorecipe.genai.GenAIChat import GenAIChatClient
import uuid
import mlflow
from dotenv import load_dotenv
import uuid
import mlflow
import pandas as pd
from validation import extract_things_from_string, calculate_precision, calculate_recall
import pandas as pd
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationReturnOptions,
)

def get_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith("genai_context_docs"):
            csv_files.append(os.path.join(directory, filename))
    return csv_files

creds = {
    "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
    "api_endpoint": "https://bam-api.res.ibm.com",
}

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v0-1-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
]

model_id = 3
DEFAULT_CONFIG = {
    "model": LLMsets[model_id],
    "params": {
        "decoding_method": DecodingMethod.GREEDY,
        "min_new_tokens": 200,
        "max_new_tokens": 3000,  # 1500,
        "stop_sequences": ["(TOKENSTOP)"],
        "return_options": TextGenerationReturnOptions(
            input_text=False, input_tokens=True
        ),
        "moderations": ModerationParameters(
            hap=ModerationHAP(input=True, output=False, threshold=0.01)
        ),
    },
    "creds": {
        "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
        "api_endpoint": "https://bam-api.res.ibm.com",
    },
    "stream": True,
}

load_dotenv()
api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_url = "https://bam-api.res.ibm.com"

SystemPrompt = """
Your task is to come up with a short plan to help me accomplish my goal in four steps using 
provided taxonomy. You can take the help of taxonomy below to create new plan. 

Taxonomy can be understood as follows: Failure Mode, Failure Definition, Failure Description, where 
failure mode is a class label used to indicate that the equipment has experienced a problem/failure, 
failure definition is a one-two word brief description of the failure, and failure 
description is a more detailed description of the failure.   

Here is failure mode taxonomy:

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

Goal: generate background document for finding failure locations based on asset class and asset description.

Asset Class: Valve - Steam Turbine - Steam Stop Valve Balanced Type using a Single-Acting Actuator

Asset Description: A steam turbine-driven balanced type stop valve with a single-acting actuator is a crucial component in 
steam turbine systems. Its primary function is to regulate or halt the flow of steam to the turbine. 
Here are the main components of this valve:

1. Valve Chest: The valve chest is the primary body of the valve, which includes valve caps, studs, and nuts. 
It provides a protective housing for the internal components of the valve.
2. Strainer: The strainer is a device installed upstream of the valve to remove any impurities or debris from 
the steam before it enters the valve.
3. Valve Disk with Bypass Stem: The valve disk is a circular plate attached to the valve stem, which moves in 
response to the actuator. The bypass stem is a small rod attached to the valve disk, allowing a small amount 
of steam to bypass the valve disk when it is in the closed position.
4. Seat, Bushings, and Packing Gland: The seat is a flat or conical surface that the valve disk presses against 
to stop the flow of steam. The bushings and packing gland provide a seal between the valve stem and the valve 
body to prevent steam from leaking.
5. Valve Seat in Body: The valve seat is a critical component that provides a seal between the valve disk and 
the valve body. It is typically made of a hard, wear-resistant material such as stainless steel.
6. Valve to Actuator Coupling: The valve to actuator coupling is a mechanical device that connects the valve to 
the actuator. It allows for the actuator to open and close the valve by moving the valve stem.
7. Single-Acting Actuator: The actuator is a mechanical device that converts energy into motion. In this case, 
it is a single-acting actuator, which uses energy to move the valve stem in one direction.
8. Pressure Balancing System: The balanced type valve has a pressure balancing system, which maintains a 
consistent force on the valve disk regardless of the pressure difference across the valve. This system ensures 
that the valve disk moves smoothly and evenly, reducing wear and tear on the valve components.
9. Positioner: The positioner is a device that controls the actuator's position based on the input signal. 
It adjusts the actuator's output to achieve the desired valve position.

# Think: The asset description lists various components, sub-components and its functionality. Read the asset 
description carefully and generates comprehensive information for components and sub-components. 

Step 1: Identify components and sub-components

1. Valve Chest:
	+ Valve Cover: A protective cover that encloses the valve components and provides access for maintenance and inspection.
	+ Valve Body: The main body of the valve that contains the internal components, including the valve seat, valve disk, and valve stem.
	+ Valve Stem: A rod that connects the valve disk to the actuator, allowing for linear motion to open and close the valve.
	+ Valve Cap: A protective cover that encloses the valve stem and provides a seal to prevent steam leakage.
	+ Studs and Nuts: Fasteners that secure the valve cover, valve body, and valve stem to the valve chest.
2. Strainer:
	+ Strainer Basket: A mesh basket that traps impurities and debris from the steam before entering the valve.
	+ Strainer Body: The main body of the strainer that contains the strainer basket and connects to the piping.
	+ Strainer Cover: A protective cover that encloses the strainer basket and provides access for maintenance and inspection.
3. Valve Disk with Bypass Stem:
	+ Valve Disk: A circular plate attached to the valve stem that moves in response to the actuator to regulate or halt the flow of steam.
	+ Bypass Stem: A small rod attached to the valve disk that allows a small amount of steam to bypass the valve disk when it is in the closed position.
4. Seat, Bushings, and Packing Gland:
	+ Seat Ring: A ring that provides a seal between the valve disk and valve body.
	+ Bushings: Components that provide a seal between the valve stem and valve body.
	+ Packing Gland: A component that provides a seal between the valve stem and valve body to prevent steam leakage.
5. Valve Seat in Body:
	+ Valve Seat Insert: A replaceable insert that provides a seal between the valve disk and valve body.
	+ Valve Seat Retainer: A component that holds the valve seat insert in place.
6. Valve to Actuator Coupling:
	+ Coupling Rod: A rod that connects the valve stem to the actuator, allowing for linear motion to open and close the valve.
	+ Coupling Bracket: A bracket that secures the coupling rod to the valve stem and actuator.
7. Single-Acting Actuator:
	+ Actuator Body: The main body of the actuator that contains the mechanical components.
	+ Actuator Piston: A piston that converts energy into linear motion to open and close the valve.
	+ Actuator Spring: A spring that provides a force to return the actuator piston to its original position.
	+ Actuator Diaphragm: A flexible membrane that separates the actuator piston from the fluid or gas medium.
8. Pressure Balancing System:
	+ Balancing Line: A line that connects the high-pressure and low-pressure sides of the valve to maintain a consistent force on the valve disk.
	+ Balancing Valve: A valve that regulates the flow of fluid or gas in the balancing line.
9. Positioner:
	+ Positioner Body: The main body of the positioner that contains the mechanical and electrical components.
	+ Positioner Sensor: A sensor that detects the position of the valve and provides feedback to the positioner.
	+ Positioner Controller: A controller that adjusts the actuator output to achieve the desired valve position.

# Think: Use failure mode taxonomy as guideline and generates all the failure modes for components and sub-components. 
Also include components/subcomponents that are not listed in asset description above but may be associated with 
failure modes. Include electrical, mechanical, thermal and others factors that affect failure mode if any.

Step 2: Identify Failure Modes for Each Component

1. Valve Chest:
	+ BUR: Overheating of valve chest leading to burning and consumption of the material.
	+ DFM: Deformation of valve chest due to excessive pressure or temperature.
	+ FRC: Cracking or fracturing of valve chest due to fatigue or overload.
	+ FTO: Failure to open or close due to mechanical or electrical issues with the actuator or valve stem.
	+ STP: Does not stop on demand due to mechanical or electrical issues with the actuator or valve stem.
	+ STU: Becoming stuck due to mechanical or electrical issues with the actuator or valve stem.
2. Strainer:
	+ CRK: Cracking of strainer basket or body due to fatigue or overload.
	+ MBU: Material build-up in strainer basket or body due to impurities or debris in the steam.
	+ INL: Internal leakage due to corrosion or wear of strainer body or gasket.
	+ ELK: External leakage due to damage or wear of strainer body or gasket.
3. Valve Disk with Bypass Stem:
	+ BUR: Overheating of valve disk leading to burning and consumption of the material.
	+ FRC: Cracking or fracturing of valve disk due to fatigue or overload.
	+ SLP: Slippage of valve disk due to wear or corrosion of the valve seat or disk.
	+ PDE: Parameter deviation of valve disk position due to mechanical or electrical issues with the actuator or positioner.
4. Seat, Bushings, and Packing Gland:
	+ BUR: Overheating of valve seat or bushings leading to burning and consumption of the material.
	+ RUS: Rust formation on valve seat or bushings due to moisture or condensation.
	+ INL: Internal leakage due to corrosion or wear of valve seat or bushings.
	+ ELK: External leakage due to damage or wear of valve seat or bushings.
5. Valve Seat in Body:
	+ BUR: Overheating of valve seat insert leading to burning and consumption of the material.
	+ FRC: Cracking or fracturing of valve seat insert due to fatigue or overload.
	+ TRD: Thickness reduction of valve seat insert due to wear or corrosion.
6. Valve to Actuator Coupling:
	+ FTO: Failure to open or close due to mechanical or electrical issues with the coupling rod or bracket.
	+ STP: Does not stop on demand due to mechanical or electrical issues with the coupling rod or bracket.
	+ STU: Becoming stuck due to mechanical or electrical issues with the coupling rod or bracket.
7. Single-Acting Actuator:
	+ PDE: Parameter deviation of actuator output due to mechanical or electrical issues with the actuator body, piston, spring, or diaphragm.
	+ PTF: Power or signal failure due to electrical or mechanical issues with the actuator or positioner.
	+ SAL: Spurious alarms due to electrical or mechanical issues with the actuator or positioner.
8. Pressure Balancing System:
	+ PDE: Parameter deviation of balancing line pressure due to mechanical or electrical issues with the balancing valve or line.
	+ PTF: Power or signal failure due to electrical or mechanical issues with the balancing valve or line.
9. Positioner:
	+ PDE: Parameter deviation of valve position due to mechanical or electrical issues with the positioner body, sensor, or controller.
	+ PTF: Power or signal failure due to electrical or mechanical issues with the positioner or actuator.
	+ SAL: Spurious alarms due to electrical or mechanical issues with the positioner or actuator.

# Think: Generate failure modes for additional sub-compoents associated with the given asset

Step 3: Identify Failure Modes for Additional Components and Subcomponents

1. Pressure Vessels:
	+ BUR: Overheating of pressure vessels leading to burning and consumption of the material.
	+ DFM: Deformation of pressure vessels due to excessive pressure or temperature.
	+ FRC: Cracking or fracturing of pressure vessels due to fatigue or overload.
	+ ELK: External leakage due to damage or wear of pressure vessel body or gasket.
	+ TRD: Thickness reduction of pressure vessels due to wear or corrosion.
2. Piping and Fittings:
	+ CRK: Cracking of piping or fittings due to fatigue or overload.
	+ MBU: Material build-up in piping or fittings due to impurities or debris in the steam.
    + INL: Internal leakage due to corrosion or wear of piping or fittings.
    + ELK: External leakage due to damage or wear of piping or fittings.
    + TRD: Thickness reduction of piping or fittings due to wear or corrosion.
3. Sensors and Transmitters:
    + PDE: Parameter deviation of sensor or transmitter output due to mechanical or electrical issues.
    + PTF: Power or signal failure due to electrical or mechanical issues with the sensor or transmitter.
    + SAL: Spurious alarms due to electrical or mechanical issues with the sensor or transmitter.
4. Insulation:
    + RUS: Rust formation on insulation due to moisture or condensation.
    + MBU: Material build-up on insulation due to impurities or debris in the steam
5. Support Structure:
    + DFM: Deformation of support structure due to excessive load or vibration.
    + FRC: Cracking or fracturing of support structure due to fatigue or overload.
6. Fasteners and Bolting:
    + CRK: Cracking of fasteners or bolting due to fatigue or overload.
    + FRC: Fracturing of fasteners or bolting due to overload or corrosion.
7. Electrical Components:
    + PDE: Parameter deviation of electrical component output due to mechanical or electrical issues.
    + PTF: Power or signal failure due to electrical or mechanical issues with the electrical component.
    + SAL: Spurious alarms due to electrical or mechanical issues with the electrical component.

# Think: Given the Step 1-3, generate a list of failure locations. Generate answer in numbered list.

Step 4: Identify Failure Locations

1. Valve Cover
2. Valve Body
3. Valve Stem
4. Valve Cap
5. Studs and Nuts
6. Strainer Basket
7. Strainer Body
8. Strainer Cover
9. Valve Disk
10. Bypass Stem
11. Seat Ring
12. Bushings
13. Packing Gland
14. Valve Seat Insert
15. Valve Seat Retainer
16. Coupling Rod
17. Coupling Bracket
18. Actuator Body
19. Actuator Piston
20. Actuator Spring
21. Actuator Diaphragm
22. Balancing Line
23. Balancing Valve
24. Positioner Body
25. Positioner Sensor
26. Positioner Controller
28. Pressure Vessels
29. Piping and Fittings
30. Sensors and Transmitters
31. Insulation
32. Support Structure
33. Fasteners and Bolting
34. Electrical Components

# Think: To succeed, I need to perform all these steps, one after the other. So I need to use the "AND" operator.
Execution Order: (Step 1 AND Step 2 AND Step 3 AND Step 4). Goal completed!

Your task is to come up with a short plan to help me accomplish my goal in four steps using 
provided taxonomy. You can take the help of taxonomy below to create new plan. Keep in mind that:
- Do not update taxonomy.
- You cannot repeat same Step in plan.
- Do not add any new steps.
- Do not modify the existing steps.

"""

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v01-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
]

params = {
    "decoding_method": DecodingMethod.GREEDY,
    "min_new_tokens": 100,
    "max_new_tokens": 4000,  # 1500,
    "stop_sequences": ["(TOKENSTOP)"],
    "return_options": TextGenerationReturnOptions(input_text=False, input_tokens=True),
    "moderations": ModerationParameters(
        hap=ModerationHAP(input=True, output=False, threshold=0.01)
    ),
}

experiment_name = "MyExperiment_" + str(uuid.uuid4())
experiment_id = mlflow.create_experiment(experiment_name)
model_id = 3

qpromt = """
Goal: generate background document for finding failure locations based on asset class and asset description.

Asset Class: {assetclass}

Asset Description: {assetdesc}
"""

assetclass = "Valve - Steam Turbine - Steam Stop Valve Balanced Type using a Single-Acting Actuator"

assetdesc = """The Steam Turbine Balanced Type Stop Valve with Single-Acting Actuator is a critical component in the operation of a steam turbine system. 
The valve chest, strainer, valve disk with bypass stem, and seat, bushings, and packing gland are the main components of this valve system. 
The valve chest can experience leakage due to wear and tear of valve caps, studs, and nuts, leading to reduced 
valve performance and potential damage to the valve and turbine. The strainer can become clogged due to buildup 
of impurities or debris, leading to reduced valve performance and potential damage to the valve and turbine. 
The valve disk with bypass stem can experience leakage due to wear and tear of the valve disk, stem, or packing, 
and potential failure due to overloading, excessive pressure, or corrosion. The seat, bushings, and packing 
gland can also experience leakage due to wear and tear, leading to reduced valve performance and potential 
damage to the valve and turbine. Failure of any of these components can lead to potential failure of the valve. 
Regular maintenance and inspection of these components can help prevent failures and ensure the safe and efficient 
operation of the steam turbine system.

"""

print (SystemPrompt)
print (qpromt.format(assetclass=assetclass, assetdesc=assetdesc))
exit(0)

tmpClient = GenAIChatClient(name='Testing',
                                description='I am testing classifier',
                                skill='Generic',
                                model=LLMsets[model_id],
                                params=params,
                                credentials=creds,
                                system_message=SystemPrompt,
                                stateful=False)

answer = tmpClient.create(
        context=None,
        messages=[{"content": qpromt.format(assetclass=assetclass, assetdesc=assetdesc), "role": "user"}],
        experiment_id=experiment_id,
    )
print (answer)