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

sample_assetclass = "CNC Robotic Containerization System"

sample_assetdesc = """The CNC Robotic Containerization System is a highly automated system used for sorting and placing various types of containers, boxes, and other 
materials onto pre-designated pallets or bins for shipment. The system includes gantry robots, docking stations, system consoles, a control panel, 
and a keyboard, each of which has the potential for failure. Failure modes for the gantry robots include reduced accuracy, reduced speed, failure 
to move, incorrect movement, failure to pick up containers, and failure to place containers. Docking stations can fail to hold containers, pallets, or 
bins, fail to release containers, pallets, or bins, or experience communication failures. System consoles can fail to provide real-time information, 
fail to monitor or control operations, or fail to display information. The control panel can experience failures in the emergency stop pushbutton, 
switches, starting or stopping the system, or interfacing with the computer. The keyboard can fail to interface with the computer, control operations, 
or monitor operations. Failure effects include reduced efficiency, reduced productivity, increased downtime, increased labor costs, increased material 
handling costs, increased risk of damage to containers, pallets, and bins, reduced safety, and increased risk of accidents."," The CNC Robotic 
Containerization System is designed for precise placement of containers on pre-designated pallets or bins, with potential failure consequences 
including misaligned containers and inefficient use of space. The system's software has potential failure modes such as bugs or glitches that 
can impact customization and configuration, while the modular design allows for expansion and upgrades with potential failure modes including 
compatibility issues and communication breakdowns. The system consoles have potential failure causes such as hardware issues that can affect 
system monitoring and control, and the docking stations ensure stable platform use through mechanical components such as bearings and gears 
that are prone to failure due to wear and tear or lack of maintenance. The keyboard plays a role in system control and monitoring, with potential failure 
modes including malfunction or damage that can impact operation, and the emergency stop pushbutton and switches function to halt system operations in 
case of emergency, with potential failure consequences including system damage or injury. The system handles container sorting through mechanical 
and software components that are prone to failure due to wear and tear, lack of maintenance, or design issues, and safety and security measures are 
in place for the emergency stop pushbutton and switches, including regular testing and maintenance. The system monitors and controls real-time information
 through software components that are prone to failure due to software bugs or design issues, and docking station failure modes include mechanical 
failure or communication breakdown that can impact system performance. The gantry robots ensure safe handling and transportation through mechanical 
components that are prone to failure due to wear and tear, lack of maintenance, or design issues, and coordination failure modes include communication 
breakdown or mechanical failure that can impact system performance. The maintenance and inspection programs for the gantry robots and docking stations 
include regular lubrication, inspection, and replacement of worn-out parts, and predictive maintenance programs use data analysis to predict and prevent 
failures. The system consoles monitor and alert to failures through software components that are prone to failure due to software bugs or design issues, 
and software updates and patches are managed through testing and validation procedures. The training and expertise required for maintaining and 
troubleshooting the gantry robots and docking stations include knowledge of mechanical and software components, and procedures for responding to 
failures include corrective actions and root cause analysis. The environmental conditions in which the system operates, such as temperature and humidity, 
can impact system performance through potential failure modes such as.
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

    answer = answer.split('\n\n')[0]
    components = extract_components(answer)
    return (answer, components)

ans = get_components(sample_assetclass, sample_assetdesc, model_id=2)
print (ans)