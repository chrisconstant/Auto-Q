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
    "ibm-mistralai/mixtral-8x7b-instruct-v01-q",
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
        "max_new_tokens": 4000,  # 1500,
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
Your task is to come up with a short plan to help me accomplish my goal in few steps using 
provided taxonomy. You can take the help of taxonomy below to create new plan. 

Taxonomy can be understood as follows: Failure Mode, Failure Description, where 
failure mode is a class label used to indicate that the equipment has experienced a problem/failure, 
and failure description is a more detailed description of the failure.   

Here is failure mode taxonomy:

1. Burned
2. Cracked
3. Deformed
4. External leak
5. Fractured
6. Failure to close
7. Failure to open
8. Failure to start/function on demand
9. Internal leak
10. Lubrication/Fluid Sampling
11. Material build-up
12. Noise
13. Parameter deviation
14. Plugged
15. Power/signal failure
16. Rust
17. Spurious Alarm
18. Slippage
19. Does not stop on demand
20. Stuck
21. Loss of Explosion Protection (EX) Integrity
22. Thickness reduction
23. Vibration

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
	+ Valve Cover
	+ Valve Body
	+ Valve Stem
	+ Valve Cap
	+ Studs and Nuts
2. Strainer:
	+ Strainer Basket
	+ Strainer Body
	+ Strainer Cover
3. Valve Disk with Bypass Stem:
	+ Valve Disk
	+ Bypass Stem
4. Seat, Bushings, and Packing Gland:
	+ Seat Ring
	+ Bushings
	+ Packing Gland
5. Valve Seat in Body:
	+ Valve Seat Insert
	+ Valve Seat Retainer
6. Valve to Actuator Coupling:
	+ Coupling Rod
	+ Coupling Bracket
7. Single-Acting Actuator:
	+ Actuator Body
	+ Actuator Piston
	+ Actuator Spring
	+ Actuator Diaphragm
8. Pressure Balancing System:
	+ Balancing Line
	+ Balancing Valve
9. Positioner:
	+ Positioner Body
	+ Positioner Sensor
	+ Positioner Controller

# Think: Use failure mode taxonomy as guideline and generates all the failure modes for components and sub-components. 
Also include components/subcomponents that are not listed in asset description above but may be associated with 
failure modes. Include electrical, mechanical, thermal and others factors that affect failure mode if any.

Step 2: Identify Failure Modes for Each Component

1. Valve Chest:
	+ Overheating, Deformation, Cracking, Failure to open
2. Strainer:
	+ Cracking, Material build-up, Internal leakage, External leakage
3. Valve Disk with Bypass Stem:
	+ Overheating, Cracking, Slippage, Parameter deviation
4. Seat, Bushings, and Packing Gland:
	+ Overheating, Rust formation, Internal leakage, External leakage
5. Valve Seat in Body:
	+ Overheating, Cracking, Thickness reduction
6. Valve to Actuator Coupling:
	+ Failure to open or close, Does not stop, stuck
7. Single-Acting Actuator:
	+ Parameter deviation, Power or signal failure
8. Pressure Balancing System:
	+ Parameter deviation, Power or signal failure
9. Positioner:
	+ Parameter deviation, Power or signal failure

# Think: Generate failure modes for additional components. This Step is optional.

Step 3: (Optional) Identify Failure Modes for Additional Components

1. Pressure Vessels:
	+ Overheating, Deformation, Cracking
2. Piping and Fittings:
	+ Cracking, Material build-up
3. Sensors and Transmitters:
    + Parameter deviation, Power or signal failure
4. Insulation:
    + Rust formation
5. Support Structure:
    + Deformation
6. Fasteners and Bolting:
    + Cracking, Fracturing
7. Electrical Components:
    + Parameter deviation

# Think: We now generate a list of all failure locations using Step 1-3.

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

# Think: Failure Locations are generated. Goal completed!

Your task is to come up with a short plan to help me accomplish my goal in few steps using 
provided taxonomy. You can take the help of taxonomy below to create new plan. Keep in mind that:
- Do not update taxonomy.
- You cannot repeat same Step in plan.

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
model_id = 2

qpromt = """Goal: generate background document for finding failure locations based on asset class and asset description.

Asset Class: {assetclass}

Asset Description: {assetdesc}
"""

assetclass = "Current Transformer"

assetdesc = """A Current Transformer (CT) is a type of transformer that is used to measure the current in an electrical 
circuit. It works by producing a secondary current that is proportional to the primary current, allowing for 
the measurement of high currents in a safe and manageable way. The CT-15kV model is designed to measure 
currents in high voltage circuits up to 15kV.

The main components of a CT include the core, the primary winding, the secondary winding, and the wiring. 
The core is typically made of a ferromagnetic material, such as iron or steel, and is used to concentrate the 
magnetic field produced by the primary winding. The primary winding is the conductor through which the current 
to be measured flows. The secondary winding is the conductor that produces the secondary current, which is 
proportional to the primary current. The wiring refers to the connections between the primary and secondary 
windings, as well as the connections to the metering instrumentation.

In this specific case, the CT-15kV is directly associated with the CT bus or cable, which is the point where 
the primary current is measured. The CT bus or cable is a critical component of the CT as it provides the path 
for the primary current to flow through the CT. The local wiring leading to the metering instrumentation refers 
to the wiring that connects the CT to the instrumentation that will measure and display the current. This 
wiring is typically designed to be highly accurate and reliable, with low levels of noise and interference, 
in order to ensure accurate measurements.

Additionally, the CT-15kV may also include other components such as a protective casing or enclosure, which is 
designed to protect the CT from environmental factors such as moisture, dust, and temperature fluctuations. 
The casing may also provide a means of mounting the CT to a structure or equipment. The CT may also include a 
burial kit, which is used to bury the CT underground in certain applications. The burial kit typically includes 
a conduit and a sealing system to protect the CT from moisture and other environmental factors.

It is important to note that the CT-15kV must be properly calibrated and maintained in order to ensure accurate 
measurements. Regular testing and maintenance can help to ensure that the CT is operating within the specified 
tolerances and that the measurements are accurate. 
"""

print (SystemPrompt)
print (qpromt.format(assetclass=assetclass, assetdesc=assetdesc))

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