from autorecipe.genai.GenAIChat import GenAIChatClient
import uuid
import mlflow
from dotenv import load_dotenv
import uuid
import mlflow
import os
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationReturnOptions,
)
import re

def extract_components(text):
    components = []
    current_component = None

    # Regular expression to match component names and subcomponents
    component_pattern = re.compile(r'^\d+\.\s*(.*):$')
    subcomponent_pattern = re.compile(r'^\s*\+\s*(.*):')

    lines = text.split('\n')
    for line in lines:
        # Check if the line matches a component name
        component_match = component_pattern.match(line)
        if component_match:
            component_name = component_match.group(1).strip()
            current_component = {"name": component_name, "subcomponents": []}
            components.append(current_component)
        else:
            # Check if the line matches a subcomponent
            subcomponent_match = subcomponent_pattern.match(line)
            if subcomponent_match and current_component is not None:
                subcomponent_name = subcomponent_match.group(1).strip()
                current_component["subcomponents"].append(subcomponent_name)

    return components


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
The asset description lists various components, sub-components and its functionality. 
Your task is to read the asset description carefully and generates comprehensive 
information for components and sub-components using provided asset description. 
You can take a look at the following examplers. Output should contain only generate Answer.

Question: Generate comprehensive information for components and sub-components. They are named entities.

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

Answer: Here is the list of components and subcomponents.

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
    "max_new_tokens": 3000,  # 1500,
    "stop_sequences": ["(TOKENSTOP)"],
    "return_options": TextGenerationReturnOptions(input_text=False, input_tokens=True),
    "moderations": ModerationParameters(
        hap=ModerationHAP(input=True, output=False, threshold=0.01)
    ),
}

qpromt = """
Question: Generate comprehensive information for components and sub-components. They are named entities.

Asset Class: {assetclass}

Asset Description: {assetdesc}

Answer: Here is the list of components and subcomponents.

"""

sample_assetclass = "Battery Charger"

sample_assetdesc = """
 Sure, here's a more comprehensive description of a Battery Charger asset and its components, including the component boundary of the battery charger input breakers:

A battery charger is an industrial asset that is used to charge batteries for various applications such as backup power systems, electric vehicles, and renewable energy systems. The main components of a battery charger include the following:

1. Input Breakers: These are electrical switches that control the flow of electrical current from the power source to the charger. They are designed to protect the charger and the battery from overcurrent and short-circuit conditions. The input breakers are an essential component of the battery charger asset, as they ensure the safe and reliable operation of the charger.
2. DC-DC Converter: This component converts the input voltage and current from the power source to the required voltage and current for charging the battery. It is typically a high-efficiency converter that minimizes energy loss and heat generation. The DC-DC converter is a critical component of the battery charger, as it determines the charging voltage and current that are applied to the battery.
3. Battery Interface: This component connects the charger to the battery and manages the flow of electrical current during charging and discharging. It also monitors the battery's state of charge, voltage, and temperature. The battery interface is a crucial component of the battery charger, as it ensures the safe and efficient transfer of electrical energy between the charger and the battery.
4. Control and Monitoring System: This component manages the operation of the charger and monitors its performance. It receives input from sensors and sends output to actuators to control the charging process. It also provides alarm and warning signals in case of faults or abnormal conditions. The control and monitoring system is a vital component of the battery charger, as it ensures the safe and reliable operation of the charger and provides real-time monitoring of the charging process.
5. Cooling System: This component is responsible for dissipating the heat generated by the charger during operation. It can be a fan, a heat sink, or a liquid cooling system, depending on the size and rating of the charger. The cooling system is an essential component of the battery charger, as it ensures the safe operation of the charger and prevents overheating, which can damage the charger's internal components.
6. Housing and Enclosure: This component protects the charger's internal components from environmental factors such as dust, moisture, and mechanical stress. It also provides a safe and convenient way to install and maintain the charger. The housing and enclosure are important components of the battery charger, as they provide a protective barrier against environmental factors and ensure the safe operation of the charger.
7. Electrical Interconnects: These components connect the charger to the power source, battery, and other peripheral devices. They are designed to handle the electrical current and voltage requirements of the charger and battery. The electrical interconnects are critical components of the battery charger, as they ensure the safe and efficient transfer of electrical energy between the charger and the power source and battery.
8. Insulation and Grounding: These components ensure the safe operation of the charger by providing electrical insulation and grounding between the charger's internal components and the external environment. Proper insulation and grounding are essential for the safe operation of the charger and prevent electrical shock or damage to the charger's internal components. 
"""

def get_components(assetclass, assetdesc, model_id=3):

    experiment_name = "MyExperiment_" + str(uuid.uuid4())
    experiment_id = mlflow.create_experiment(experiment_name)

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

    answer = answer.strip()
    answer_split = answer.split('\n\n')
    if len(answer_split) == 2:
        if len(answer_split[0]) > len(answer_split[1]):
            answer = answer_split[0]
        else:
            answer = answer_split[1]
    else:
        pass

    components = extract_components(answer)
    return (answer, components)

#ans = get_components(sample_assetclass, sample_assetdesc, model_id=1)
#print (ans)