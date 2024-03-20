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


def extract_components(paragraph):
    components = []
    current_component = None

    lines = paragraph.split('\n')
    for line in lines:
        # If the line starts with a digit followed by a dot, it's a component name
        if line.strip() and line[0].isdigit():
            component_name = line.strip().split('.')[1].strip()
            current_component = {"name": component_name, "subcomponents": []}
            components.append(current_component)
        # If the line starts with '+', it's a subcomponent
        elif current_component is not None and line.strip().startswith('+'):
            subcomponent_name = line.strip()[1:].strip()  # Remove the '+' sign
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
Your task is to generate failure locations from the failure modes. You need to carefully read the failure 
mode description as it also mentioned certain sub-components. Please also include sub-components into 
final answer. Generate only answer.

Question: Identify all components/subcomponents that are mentioned in Failure Mode Description and can be used 
as failure locations. Failure location are named entity representing a location on an asset. Generate one line per 
failure locations. Include failure locations at component level, subcomponent level, and so on. Only generate list of
failure locations. Do not append component name to sub-component name.

Failure Mode Description: In the following we provided a failure mode for each component and subcomponents.

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

Answer:

1. Valve Chest
	+ Valve Chest Body
	+ Valve Chest Cover
	+ Valve Chest Internals
2. Strainer
	+ Strainer Body
	+ Strainer Basket
	+ Strainer Gasket
3. Valve Disk with Bypass Stem
	+ Valve Disk
	+ Valve Disk Bypass Stem
	+ Valve Disk Seal
4. Seat, Bushings, and Packing Gland
	+ Valve Seat
	+ Valve Bushings
	+ Packing Gland
5. Valve Seat in Body
	+ Valve Seat Insert
	+ Valve Seat Body
6. Valve to Actuator Coupling
	+ Coupling Rod
	+ Coupling
7. Single-Acting Actuator
	+ Actuator Body
	+ Actuator Piston
	+ Actuator Spring
	+ Actuator Diaphragm
8. Pressure Balancing System
	+ Balancing Valve
	+ Balancing Line
	+ Pressure Sensor
9. Positioner
	+ Positioner Body
	+ Positioner Sensor
	+ Positioner Controller

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
Question: Identify all components/subcomponents that are mentioned in Failure Mode Description and can be used 
as failure locations. Failure location are named entity representing a location on an asset. Generate one line per 
failure locations. Include failure locations at component level, subcomponent level, and so on. Only generate list of
failure locations. Do not append component name to sub-component name.

Failure Mode Description: In the following we provided a failure mode for each component and subcomponents.

{assetdesc}

Answer:

"""

assetdesc = """
1. Gantry Robots:
        + BUR: Overheating of the gantry robot arm leading to burning and consumption of the material.
        + CRK: Cracking or fracturing of the gantry robot arm due to fatigue or overload.
        + DFM: Deformation of the gantry robot arm due to excessive pressure or temperature.
        + ELK: External leakage due to damage or wear of the gantry robot arm.
        + INL: Internal leakage due to corrosion or wear of the gantry robot arm.
        + LFS: Unsatisfactory fluid/lubrication sample in the drive system.
        + PTF: Power or signal failure in the control system.
        + RUS: Rust formation on the robot arm due to moisture or condensation.
        + SAL: Spurious alarms in the control system.
2. Docking Stations:
        + BUR: Overheating of the docking station frame leading to burning and consumption of the material.
        + CRK: Cracking or fracturing of the docking station frame due to fatigue or overload.
        + DFM: Deformation of the docking station frame due to excessive pressure or temperature.
        + ELK: External leakage due to damage or wear of the docking station frame.
        + INL: Internal leakage due to corrosion or wear of the docking station frame.
        + LFS: Unsatisfactory fluid/lubrication sample in the mechanical components.
        + PTF: Power or signal failure in the electrical components.
        + RUS: Rust formation on the docking station frame due to moisture or condensation.
        + SAL: Spurious alarms in the electrical components.
3. System Consoles:
        + BUR: Overheating of the system console frame leading to burning and consumption of the material.
        + CRK: Cracking or fracturing of the system console frame due to fatigue or overload.
        + DFM: Deformation of the system console frame due to excessive pressure or temperature.
        + ELK: External leakage due to damage or wear of the system console frame.
        + INL: Internal leakage due to corrosion or wear of the system console frame.
        + LFS: Unsatisfactory fluid/lubrication sample in the control systems.
        + PTF: Power or signal failure in the control systems.
        + RUS: Rust formation on the system console frame due to moisture or condensation.
        + SAL: Spurious alarms in the control systems.
4. Control Panel:
        + BUR: Overheating of the control panel frame leading to burning and consumption of the material.
        + CRK: Cracking or fracturing of the control panel frame due to fatigue or overload.
        + DFM: Deformation of the control panel frame due to excessive pressure or temperature.
        + ELK: External leakage due to damage or wear of the control panel frame.
        + INL: Internal leakage due to corrosion or wear of the control panel frame.
        + LFS: Unsatisfactory fluid/lubrication sample in the emergency stop pushbutton and switches.
        + PTF: Power or signal failure in the indicator lights and control systems.
        + RUS: Rust formation on the control panel frame due to moisture or condensation.
        + SAL: Spurious alarms in the emergency stop pushbutton and switches.
5. Keyboard:
        + BUR: Overheating of the keyboard frame leading to burning and consumption of the material.
        + CRK: Cracking or fracturing of the keyboard frame due to fatigue or overload.
        + DFM: Deformation of the keyboard frame due to excessive pressure or temperature.
        + ELK: External leakage due to damage or wear of the keyboard frame.
        + INL: Internal leakage due to corrosion or wear of the keyboard frame.
        + LFS: Unsatisfactory fluid/lubrication sample in the keys, switches, or control systems.
        + PTF: Power or signal failure in the keys, switches, or control systems.
        + RUS: Rust formation on the keyboard frame due to moisture or condensation.
        + SAL: Spurious alarms in the keys, switches, or control systems.
"""

def get_failure_locations(assetdesc, model_id):
    """_summary_
    """
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
            messages=[{"content": qpromt.format(assetdesc=assetdesc), "role": "user"}],
            experiment_id=experiment_id,
        )

    answer = answer.split('\n\n')[0]
    components = extract_components(answer)
    return components
