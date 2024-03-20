from autorecipe.genai.GenAIChat import GenAIChatClient
import uuid
import mlflow
from dotenv import load_dotenv
import uuid
import mlflow
import os
import pandas as pd
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

load_dotenv()
api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_url = "https://bam-api.res.ibm.com"

SystemPrompt = """
Your task is to use failure mode taxonomy as guideline and generates all the failure modes for components and sub-components. 
Also include components/subcomponents that are not listed in asset description above but may be associated with 
failure modes. Include electrical, mechanical, thermal and others factors that affect failure mode if any. 

Here is failure mode taxonomy (failure mode, definition, description):

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

Question: Generate failure mode for each component and subcomponents from the asset description.

Asset Description including components and its subcomponent:

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

Answer: 

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

qpromt = """
Question: Generate failure mode for each component and subcomponents from the asset description.

Asset Description including components and its subcomponent: 

{assetdesc}

Answer:
"""

sample_assetdesc = """
1. Gantry Robots:
        + Gantry Robot Arm: The main arm of the robot that moves in three dimensions to manipulate containers.
        + End Effector: The tool or gripper attached to the end of the robot arm that interacts with the containers.
        + Drive System: The mechanical components that move the robot arm and end effector.
        + Control System: The electronic components that control the robot's movement and operation.
2. Docking Stations:
        + Docking Station Frame: The structural component that supports the other components of the docking station.
        + Containers Holders: The components that hold and secure the containers during the loading and unloading process.
        + Mechanical Components: Components such as bearings and gears that provide stability and movement to the docking station.
        + Electrical Components: Components such as sensors and wiring that enable communication and control between the docking station and the system.
3. System Consoles:
        + System Console Frame: The structural component that supports the other components of the system console.
        + Display: The visual component that provides real-time information and feedback to the operator.
        + Input Devices: Devices such as keyboards and mice that allow the operator to interact with the system.
        + Control Systems: The electronic components that control the system's operation and monitoring.
4. Control Panel:
        + Control Panel Frame: The structural component that supports the other components of the control panel.
        + Emergency Stop Pushbutton and Switches: Components that halt system operations in case of emergency.
        + Indicator Lights: Components that provide visual feedback to the operator about the system's status.
        + Control Systems: The electronic components that control the system's operation and monitoring.
5. Keyboard:
        + Keyboard Frame: The structural component that supports the other components of the keyboard.
        + Keys: The components that are pressed to input data or commands.
        + Switches: Components that enable or disable the keyboard's functions.
        + Control Systems: The electronic components that control the keyboard's operation and monitoring.
"""

def get_failure_modes(assetdesc, model_id=3):
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
            messages=[{"content": qpromt.format(assetdesc=assetdesc), "role": "user"}],
            experiment_id=experiment_id,
        )
    return answer

# get_failure_modes(sample_assetdesc)