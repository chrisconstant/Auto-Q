from autorecipe.genai.GenAIChat import GenAIChatClient
import uuid
import mlflow
from dotenv import load_dotenv
import uuid
import mlflow

from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationReturnOptions,
)


load_dotenv()
api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_url = "https://bam-api.res.ibm.com"

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v01-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
    "meta-llama/llama-3-70b-instruct",
]

final_ans = ""
final_confidence = ""

DEFAULT_CONFIG = {
    "model": LLMsets[0],
    "params": {
        "decoding_method": DecodingMethod.GREEDY,
        "min_new_tokens": 200,
        "max_new_tokens": 2048,  # 1500,
        "stop_sequences": ["(TOKENSTOP)"],
        # "stream": True,
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

def get_description(model_id = 1):
    """_summary_

    :param iteration: _description_, defaults to 10
    :type iteration: int, optional
    :param asset_class: _description_, defaults to "Electrical submersible pump"
    :type asset_class: str, optional
    """

    OriginalSysPrompt = """
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being 
    safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or 
    illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    
    If a question does not make any sense, or is not factually coherent, explain why instead of answering 
    something not correct. If you don't know the answer to a question, please don't share false information.
    """

    SystemPrompt = """
    Your task is to come up with a short plan to help me accomplish my goal in a couple of steps using 
    provided taxonomy. You can take the help of taxonomy below to create new plan.

    Taxonomy can be understood as follows: [Parent] and its [Child], where Parent is an equipment 
    name/description and Child is a list of parts/items/components associated with parent. Taxonomy is 
    non-cyclic, it means same node cannot be connected to itself. You are given an agent that can produce 
    plan by traversing given taxonomy recursively.

    Taxonomy-1:
    Parent: Battery - NICAD.
    Child:
    - Electrolyte
    - Inter Cell and Inter Tier Connectors and Hardware and Battery Cable Connectors
    - Jar & Lid
    - Plates
    - Posts
    - Rack
    - Vent or Flame Arrestor

    Taxonomy-2:
    Parent: Battery - Inverter.
    Child:
    - Capacitors, Commutation and Other Filled
    - Capacitors, Electrolytic
    - Fuse holder
    - Input / Output Filter Choke and Commutating Chokes
    - Input Fuse
    - Maintenance Bypass Switch
    - Metering Instruments (e.g. voltmeter, frequency meter, status lights)
    - Muffin fans
    - Power Semiconductor Components
    - Printed Circuit Boards (Firing circuit, Oscillator, Alarm, Sync, Metering circuits)
    - Static Switch
    - Transformer

    Taxonomy-3:
    Parent: Rectifier Bank/Charger_Switch Mode.
    Child:
    - Cooling Fans
    - Interconnecting Communication Cables
    - Interconnecting Power Cables
    - Rack and Backplane (including AC breakers)
    - Rectifier Module (including PLC, semiconductors, transformer, etc)
    - System Control Monitor (including dry contacts, circuit boards, alarms, etc)

    Goal: Generate candidate set for Battery - Charger.

    # Think: My target equipment is a Battery - Charger. I will use taxonomy to devise plan to generate 
    candidate set for equipment. 

    **Step 1: Collect the Taxonomy. **  

    We have three taxonomies: Taxonomy-1 for Battery - NICAD, Taxonomy-2 for Battery - Inverter, and Taxonomy-3 
    for Rectifier Bank/Charger_Switch Mode.

    # Think: Although the target equipment "Battery - Charger" doesn't exactly match any parent equipment in 
    the taxonomies, I can infer that it's related to the "Battery - Inverter" taxonomy, as both involve charging 
    and discharging batteries. Therefore, I'll use Taxonomy-2 to devise a plan for generating a candidate set 
    for Battery - Charger. I have used a semantic similarity between "Battery - Charger" and "Battery - Inverter" 
    to priotarize a selection.  

    **Step 2: Traverse Taxonomy-2 to generate a candidate set for Battery - Charger.**

    Using Taxonomy-2, we can generate a candidate set for Battery - Charger by traversing the child components 
    associated with the Battery - Inverter parent. Here's a possible candidate set:

    * Capacitors, Commutation and Other Filled
    * Capacitors, Electrolytic
    * Fuse holder
    * Input / Output Filter Choke and Commutating Chokes
    * Input Fuse
    * Maintenance Bypass Switch
    * Metering Instruments (e.g. voltmeter, frequency meter, status lights)
    * Muffin fans
    * Power Semiconductor Components
    * Printed Circuit Boards (Firing circuit, Oscillator, Alarm, Sync, Metering circuits)
    * Static Switch
    * Transformer

    # Think: Now I will explore other taxonomy. Since Taxonomy-3 is more closely related to charging functionality, 
    which is a primary aspect of a Battery - Charger system, it's possible that traversing this taxonomy could 
    provide additional components that are more specific to charging and switching modes, and thus help to fill 
    the gaps in the candidate set generated from Taxonomy-2.

    **Step 3: Traverse Taxonomy-3 to generate a candidate set for Battery - Charger.**

    Using Taxonomy-3, we can generate a candidate set for Battery - Charger by traversing the child components 
    associated with the Rectifier Bank/Charger_Switch Mode parent. Here's a possible candidate set:

    * Cooling Fans
    * Interconnecting Communication Cables
    * Interconnecting Power Cables
    * Rack and Backplane (including AC breakers)
    * Rectifier Module (including PLC, semiconductors, transformer, etc)
    * System Control Monitor (including dry contacts, circuit boards, alarms, etc)

    # Think: Now I will explore other taxonomy. As for Taxonomy-1, it's related to Battery - NICAD, which is a 
    different type of battery technology. Since your goal is to generate a candidate set for Battery - Charger, 
    it's unlikely that Taxonomy-1 would provide relevant components. You can safely ignore Taxonomy-1 for this 
    purpose. If Taxonomy-1 was relevant, I would have repeated Step 3 for Taxonomy-1. I complete taxonomy traversal. 

    **Step 4: Combine and Refine the Candidate Set from Step 2 and 3.

    Here is the final combined and refined candidate set for Battery - Charger:

    * Capacitors, Commutation and Other Filled
    * Capacitors, Electrolytic
    * Cooling Fans
    * Fuse holder
    * Input / Output Filter Choke and Commutating Chokes
    * Input Fuse
    * Interconnecting Communication Cables
    * Interconnecting Power Cables
    * Maintenance Bypass Switch
    * Metering Instruments (e.g. voltmeter, frequency meter, status lights)
    * Muffin fans
    * Power Semiconductor Components
    * Printed Circuit Boards (Firing circuit, Oscillator, Alarm, Sync, Metering circuits)
    * Rack and Backplane (including AC breakers)
    * Rectifier Module (including PLC, semiconductors, transformer, etc)
    * Static Switch
    * System Control Monitor (including dry contacts, circuit boards, alarms, etc)
    * Transformer

    # Think: Now that I have a comprehensive candidate set, I'll review each component to ensure it's relevant 
    to the Battery - Charger system. By removing any irrelevant components, I can further refine the candidate 
    set and increase its accuracy, without compromising the inclusion of essential components. This step will 
    help me strike a balance between comprehensiveness and precision, ultimately leading to a more effective 
    candidate set for the Battery - Charger system.

    **Step 5: Finalize the Candidate Set**

    The final combined and refined candidate set for Battery - Charger is:

    * Capacitors, Commutation and Other Filled
    * Capacitors, Electrolytic
    * Cooling Fans
    * Fuse holder
    * Input / Output Filter Choke and Commutating Chokes
    * Input Fuse
    * Interconnecting Communication Cables
    * Interconnecting Power Cables
    * Maintenance Bypass Switch
    * Metering Instruments (e.g. voltmeter, frequency meter, status lights)
    * Muffin fans
    * Power Semiconductor Components
    * Printed Circuit Boards (Firing circuit, Oscillator, Alarm, Sync, Metering circuits)
    * Rack and Backplane (including AC breakers)
    * Rectifier Module (including PLC, semiconductors, transformer, etc)
    * Static Switch
    * System Control Monitor (including dry contacts, circuit boards, alarms, etc)
    * Transformer

    # Think: Since the equipment description is 'Battery - Charger', I should consider the fundamental components 
    that are typically associated with charging systems, such as power conversion units, charging circuits, 
    control modules, and connectors, even if they are not explicitly mentioned in the provided taxonomies. 
    I will use my knowledge of charging systems to infer the presence of these components and include them in the 
    candidate set.

    **Step 6: Add Additional Components for Candidate Set**

    The final combined and refined candidate set for Battery - Charger is:

    * Capacitors, Commutation and Other Filled
    * Capacitors, Electrolytic
    * Cooling Fans
    * Fuse holder
    * Input / Output Filter Choke and Commutating Chokes
    * Input Fuse
    * Interconnecting Communication Cables
    * Interconnecting Power Cables
    * Maintenance Bypass Switch
    * Metering Instruments (e.g. voltmeter, frequency meter, status lights)
    * Muffin fans
    * Power Semiconductor Components
    * Printed Circuit Boards (Firing circuit, Oscillator, Alarm, Sync, Metering circuits)
    * Rack and Backplane (including AC breakers)
    * Rectifier Module (including PLC, semiconductors, transformer, etc)
    * Static Switch
    * System Control Monitor (including dry contacts, circuit boards, alarms, etc)
    * Transformer
    * Power conversion units
    * Charging circuits
    * Control modules
    * Connectors

    # Think: To succeed, I need to perform all these steps, one after the other. So I need to use the "AND" 
    operator. Execution Order: (Step 1 AND Step 2 AND Step 3 AND Step 4 AND Step 5 AND Step 6). Goal completed!

    Here is a different goal with different taxonomy relations. Your task is to come up with a short plan to 
    help me accomplish my goal in a couple of steps using provided taxonomy. You can take the help of taxonomy 
    below to create new plan. 
    Keep in mind that:
    - It is okay to update taxonomy than the original.
    - Be very careful with the adding of duplicate node in taxonomy.
    - You cannot use a partial taxonomy.
    - You cannot repeat same Step in plan.
    - Do not traverse newly created node in taxonomy.

    """

    UserGoal = """
    Taxonomy-1:
    Parent: Equipment Description: Bumper - Spring.
    Child:
    - Bolting
    - Bumper Head and Bracket
    - Mounting Bracket
    - Safety Cables
    - Spring
    - Spring Plunger
    - Stationary Cylinder

    Taxonomy-2:
    Parent: Equipment Description: Bumper - Solid.
    Child:
    - Bolting
    - Bumper Head and Bracket
    - Mounting Bracket
    - Safety Cables
    - Spring
    - Spring Plunger
    - Stationary Cylinder

    Taxonomy-3:
    Parent: Equipment Description: FLEX - Pump - Horizontal - Mechanical Seal Non-Oil Bath - T4 Diesel Driven.
    Child:
    - Diesel - After Cooler
    - Diesel - Air Box
    - Diesel - Alternator and Diodes
    - Diesel - Battery
    - Diesel - Battery Charger
    - Diesel - Belts and Sheaves
    - Diesel - Cam Follower Roller (if present)
    - Diesel - Camshaft, Lobes, and Bushings
    - Diesel - Connecting Rod
    - Diesel - Coolant
    - Diesel - Coolant or Block Heater
    - Diesel - Crankcase Air Breathers
    - Diesel - Crankshaft Bearings (Main, Thrust, and Connecting rod)
    - Diesel - Cylinder Head
    - Diesel - Cylinder Liners
    - Diesel - Digital Controls or ECU
    - Diesel - Electrical Devices (e.g. sensors, circuit breakers, solenoids, relays, meters, switches, fuses, push buttons, microprocessors, digital displays)
    - Diesel - Emission Control - DPF (Diesel Particulate Filter)
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Injector Nozzle & Valve
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - NOx Control Unit
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - SCR Catalyst
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Silencer
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Tubing
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Tubing & Hoses
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Tubing Heat Tracing
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Tubing Insulation
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Fluid
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Pump
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Breather
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Drain Plug
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Filter and Strainer
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Heater
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Heater Valve
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Level Float
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Temperature Control Valve & Sensor
    - Diesel - Engine Mounts
    - Diesel - Engine Valve Seats
    - Diesel - Engine Valve Springs
    - Diesel - Engine Valve Stem
    - Diesel - Engine Valve Train
    - Diesel - Exhaust Gas Recirculation (EGR) Cooler
    - Diesel - Exhaust Gas Recirculation (EGR) Valve
    - Diesel - Filter - Fuel (Pre-Fuel & Final)
    - Diesel - Filter - Inlet Air (Element or Cartridge Type)
    - Diesel - Filter - Lube Oil
    - Diesel - Fly Wheel
    - Diesel - Fuel
    - Diesel - Fuel Hoses
    - Diesel - Fuel Lines
    - Diesel - Fuel Tank
    - Diesel - Fuel Tank Breather or Vent
    - Diesel - Fuel Tank Strainer (if present)
    - Diesel - Fuel-Water Separator Element
    - Diesel - Gaskets, Seals, and O-rings (Internal and External Elastomer Type)
    - Diesel - Head Gasket
    - Diesel - High Pressure Fuel Pump
    - Diesel - Hydraulic Lifter (if present)
    - Diesel - Injector Tubing
    - Diesel - Injectors
    - Diesel - Linkages and Controls
    - Diesel - Lube Oil
    - Diesel - Lube Oil Pressure Control Device
    - Diesel - Lube Oil Pump
    - Diesel - Muffler
    - Diesel - PTO (present; not normally used in this application and is not expected to affect normal operation)
    - Diesel - Piston Wrist Pin Bearings
    - Diesel - Pistons
    - Diesel - Push Rods
    - Diesel - Radiator
    - Diesel - Radiator Cap
    - Diesel - Radiator Fan
    - Diesel - Radiator Hoses
    - Diesel - Radiator Tubing
    - Diesel - Starter
    - Diesel - Thermostat
    - Diesel - Timing Gears (if present)
    - Diesel - Turbocharger
    - Diesel - Turbocharger Exhaust Flex Hoses
    - Diesel - Turbocharger Exhaust Inlet Screen (if present)
    - Diesel - Valve Train - Rocker Arms with Rollers
    - Diesel - Vibration Damper or Harmonic Balancer
    - Diesel - Water Pump
    - Diesel - Wiring Harness
    - Diesel-Pump - Coupling - Elastomeric Element
    - Pump - Bearing Seals - Lip
    - Pump - Bearings - Rolling Element (Radial and Thrust)
    - Pump - Casing
    - Pump - Casing Drain or Stop Valve
    - Pump - Casing and Internals, if present
    - Pump - Check Valve - Disk Arm (if present)
    - Pump - Check Valve - Hinge Pin (if present)
    - Pump - Check Valve - Rubber Flapper (if present)
    - Pump - Check Valve - Seat Failure (Body or Disk)
    - Pump - Connections and Piping
    - Pump - Discharge and Suction Connections
    - Pump - Gaskets and O-Rings
    - Pump - Impeller and Wear Rings
    - Pump - Lubrication - Grease
    - Pump - Priming System
    - Pump - Seal - Mechanical Non-Oil Bath Type (process or seal water wetted)
    - Pump - Shaft
    - Skid - Diesel and Pump Base Plate or Frame
    - Trailer - Bed, Frame, and Lifting Lugs
    - Trailer - Electric Brakes
    - Trailer - Electric Lights
    - Trailer - Hitch or Coupling
    - Trailer - Hydraulic Brakes
    - Trailer - Levelers
    - Trailer - Suspension
    - Trailer - Tires
    - Trailer - Wheel Bearings
    - Trailer - Wheels Rims


    Goal: Generate candidate set for Bumper - Hydraulic.

    """
    initstep = True
    stateful = False
    asset_class = 'Test'

    experiment_name = "MyExperiment_" + asset_class + "_" + str(uuid.uuid4())
    experiment_id = mlflow.create_experiment(experiment_name)

    if initstep:
        mdl = GenAIChatClient(
            name="ADesc",
            description="Asset Description",
            skill="Generate Asset Description",
            model=LLMsets[model_id],
            params=DEFAULT_CONFIG["params"],
            system_message=OriginalSysPrompt,
            credentials=DEFAULT_CONFIG["creds"],
            stateful=stateful,
            stream=True,
        )

    sentence = SystemPrompt + UserGoal

    ans = mdl.create(
        context=None,
        messages=[{"content": sentence, "role": "user"}],
        experiment_id=experiment_id,
    )
    return ans


def get_iso_asset_description(model_id=2):

    OriginalSysPrompt = """
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being 
    safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or 
    illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    
    If a question does not make any sense, or is not factually coherent, explain why instead of answering 
    something not correct. If you don't know the answer to a question, please don't share false information.
    """

    SystemPrompt = """
    Your task is to come up with a short plan to help me accomplish my goal in a couple of steps using 
    provided ISO Standard. You can take the help of ISO Standard below to create new plan.

    ISO Standard:
    Equipment class contains comparable equipment units. For example, Heat exchangers, compressors, 
    piping, pumps, gas turbines, subsea wellhead and X-mas trees, lifeboats are example of equipment. 
    Equipment may have a subsystem necessary for the equipment unit to function. Some example of 
    subsystem includes: Lubrication subunit, cooling subunit, control and monitoring, heating subunit, 
    pelletizing subunit, quenching subunit, refrigeration subunit, reflux subunit, distributed control 
    subunit. The failure location represents the group of parts of the equipment unit that are commonly 
    maintained (repaired/ restored) as a whole. For example, Cooler, coupling, gearbox, lubrication oil pump, 
    instrument loop, motor, valve, filter, pressure sensor, temperature sensor, electric circuit are 
    example of component or maintainable item. A part is single piece of equipment such as Seal, tube, 
    shell, impeller, gasket, filter plate, bolt, nut. The failure locations are collection of components, 
    maintainable items and parts.  

    Taxonomy-1:
    Parent: Battery - NICAD.
    Child:
    - Electrolyte
    - Inter Cell and Inter Tier Connectors and Hardware and Battery Cable Connectors
    - Jar & Lid
    - Plates
    - Posts
    - Rack
    - Vent or Flame Arrestor

    Taxonomy-2:
    Parent: Battery - Inverter.
    Child:
    - Capacitors, Commutation and Other Filled
    - Capacitors, Electrolytic
    - Fuse holder
    - Input / Output Filter Choke and Commutating Chokes
    - Input Fuse
    - Maintenance Bypass Switch
    - Metering Instruments (e.g. voltmeter, frequency meter, status lights)
    - Muffin fans
    - Power Semiconductor Components
    - Printed Circuit Boards (Firing circuit, Oscillator, Alarm, Sync, Metering circuits)
    - Static Switch
    - Transformer

    Taxonomy-3:
    Parent: Rectifier Bank/Charger_Switch Mode.
    Child:
    - Cooling Fans
    - Interconnecting Communication Cables
    - Interconnecting Power Cables
    - Rack and Backplane (including AC breakers)
    - Rectifier Module (including PLC, semiconductors, transformer, etc)
    - System Control Monitor (including dry contacts, circuit boards, alarms, etc)

    Goal: Generate candidate set for Battery - Charger.

    # Think: My target equipment is a Battery - Charger. I will use ISO Standard to devise plan to generate 
    candidate set for equipment. Based on the provided ISO Standard, we discover Equipment class, subsystems,
    equipment unit/group, component or maintainable item or parts. 

    **Step 1: Identify Subsystems. **  

    For the Battery - Charger equipment, let's identify the possible subsystems necessary for it to function. 
    Some potential subsystems could be:

    * Charging subunit
    * Power conversion subunit
    * Monitoring and control subunit
    * Cooling subunit (if the charger has a cooling system)
    * Electrical protection subunit (e.g., surge protection, overvoltage protection)

    # Think: Now that I have identified the subsystems for my Battery - Charger equipment, I will think 
    about the possible failure locations within each subsystem. These failure locations will include 
    components, maintainable items, and parts that are commonly maintained or repaired as a whole. 
    For example, within the Charging subunit, I might have components like power transistors, diodes, or 
    fuses. Within the Monitoring and control subunit, I might have components like sensors, microcontrollers, 
    or displays. I will list out all these failure locations to generate a comprehensive candidate set for 
    my Battery - Charger equipment.    
    
    **Step 2: Identify Failure Locations within Subsystems**

    Within each subsystem, identify the possible failure locations, including components, maintainable items, 
    and parts that are commonly maintained or repaired as a whole. For example:

    * Charging subunit:
        + Power transistors
        + Diodes
        + Fuses
        + Transformers
    * Power conversion subunit:
        + Rectifiers
        + Inverters
        + DC-DC converters
        + Capacitors
    * Monitoring and control subunit:
        + Sensors (e.g., voltage, current, temperature)
        + Microcontrollers
        + Displays (e.g., LCD, LED)
        + Communication interfaces (e.g., USB, serial)
    * Cooling subunit (if applicable):
        + Fans
        + Heat sinks
        + Thermal sensors
        + Cooling pipes/tubes
    * Electrical protection subunit:
        + Surge protectors
        + Overvoltage protectors
        + Undervoltage protectors
        + Circuit breakers
    
    # Think: Review the failure locations identified in Step 2 and refine them by considering the 
    components and maintainable items listed in Taxonomy-1, Taxonomy-2, and Taxonomy-3.     

    **Step 3: Refine Failure Locations using Taxonomy**

    Here is the refined failure locations by considering the components and maintainable items listed in 
    Taxonomy-1, Taxonomy-2, and Taxonomy-3.
    
    * Charging subunit:
        + Power transistors
        + Diodes
        + Fuses
        + Transformers
        + Rectifier Module (from Taxonomy-3)
    * Power conversion subunit:
        + Rectifiers
        + Inverters (from Taxonomy-2)
        + DC-DC converters
        + Capacitors (from Taxonomy-2)
        + Power Semiconductor Components (from Taxonomy-2)
    * Monitoring and control subunit:
        + Sensors (e.g., voltage, current, temperature)
        + Microcontrollers
        + Displays (e.g., LCD, LED)
        + Communication interfaces (e.g., USB, serial)
        + Metering Instruments (from Taxonomy-2)
        + System Control Monitor (from Taxonomy-3)
    * Cooling subunit (if applicable):
        + Fans
        + Heat sinks
        + Thermal sensors
        + Cooling pipes/tubes
        + Cooling Fans (from Taxonomy-3)
    * Electrical protection subunit:
        + Surge protectors
        + Overvoltage protectors
        + Undervoltage protectors
        + Circuit breakers
        + Fuse holder (from Taxonomy-2)

    # Think: Since the equipment description is 'Battery - Charger', I should consider the fundamental components 
    that are typically associated with charging systems, such as power conversion units, charging circuits, 
    control modules, and connectors, even if they are not explicitly mentioned in the provided taxonomies. 
    I will use my knowledge of charging systems to infer the presence of these components and include them in the 
    candidate set.

    **Step 4: Add Additional Components for Candidate Set**

    Considering the fundamental components typically associated with charging systems, such as power 
    conversion units, charging circuits, control modules, and connectors, the additional components to 
    include in the candidate set are:

    * Charging circuits
    * Control modules
    * Connectors (e.g., power connectors, signal connectors)
    * Power conversion units (e.g., AC-DC converters, DC-DC converters)
        
    # Think: The comprehensive candidate set for the Battery - Charger equipment includes all the failure 
    locations identified in Steps 2 and 3, as well as the additional components added in Step 4.

    **Step 5: Comprehensive Candidate Set:**

    * Charging subunit:
        + Power transistors
        + Diodes
        + Fuses
        + Transformers
        + Rectifier Module
        + Charging circuits
    * Power conversion subunit:
        + Rectifiers
        + Inverters
        + DC-DC converters
        + Capacitors
        + Power Semiconductor Components
        + Power conversion units (e.g., AC-DC converters, DC-DC converters)
    * Monitoring and control subunit:
        + Sensors (e.g., voltage, current, temperature)
        + Microcontrollers
        + Displays (e.g., LCD, LED)
        + Communication interfaces (e.g., USB, serial)
        + Metering Instruments
        + System Control Monitor
        + Control modules
    * Cooling subunit (if applicable):
        + Fans
        + Heat sinks
        + Thermal sensors
        + Cooling pipes/tubes
        + Cooling Fans
    * Electrical protection subunit:
        + Surge protectors
        + Overvoltage protectors
        + Undervoltage protectors
        + Circuit breakers
        + Fuse holder
    * Connectors:
        + Power connectors
        + Signal connectors

    # Think: To succeed, I need to perform all these steps, one after the other. So I need to use the "AND" 
    operator. Execution Order: (Step 1 AND Step 2 AND Step 3 AND Step 4 AND Step 5). Goal completed!

    Here is a different goal with different taxonomy relations. Your task is to come up with a short plan to 
    help me accomplish my goal in a couple of steps using provided taxonomy. You can take the help of taxonomy 
    below to create new plan. 
    Keep in mind that:
    - It is okay to update taxonomy than the original.
    - Be very careful with the adding of duplicate node in taxonomy.
    - You cannot use a partial taxonomy.
    - You cannot repeat same Step in plan.
    - Do not traverse newly created node in taxonomy.

    """
    UserGoal = """
    Taxonomy-1:
    Parent: Equipment Description: Bumper - Spring.
    Child:
    - Bolting
    - Bumper Head and Bracket
    - Mounting Bracket
    - Safety Cables
    - Spring
    - Spring Plunger
    - Stationary Cylinder

    Taxonomy-2:
    Parent: Equipment Description: Bumper - Solid.
    Child:
    - Bolting
    - Bumper Head and Bracket
    - Mounting Bracket
    - Safety Cables
    - Spring
    - Spring Plunger
    - Stationary Cylinder

    Taxonomy-3:
    Parent: Equipment Description: FLEX - Pump - Horizontal - Mechanical Seal Non-Oil Bath - T4 Diesel Driven.
    Child:
    - Diesel - After Cooler
    - Diesel - Air Box
    - Diesel - Alternator and Diodes
    - Diesel - Battery
    - Diesel - Battery Charger
    - Diesel - Belts and Sheaves
    - Diesel - Cam Follower Roller (if present)
    - Diesel - Camshaft, Lobes, and Bushings
    - Diesel - Connecting Rod
    - Diesel - Coolant
    - Diesel - Coolant or Block Heater
    - Diesel - Crankcase Air Breathers
    - Diesel - Crankshaft Bearings (Main, Thrust, and Connecting rod)
    - Diesel - Cylinder Head
    - Diesel - Cylinder Liners
    - Diesel - Digital Controls or ECU
    - Diesel - Electrical Devices (e.g. sensors, circuit breakers, solenoids, relays, meters, switches, fuses, push buttons, microprocessors, digital displays)
    - Diesel - Emission Control - DPF (Diesel Particulate Filter)
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Injector Nozzle & Valve
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - NOx Control Unit
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - SCR Catalyst
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Silencer
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Tubing
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Tubing & Hoses
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Tubing Heat Tracing
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Tubing Insulation
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Fluid
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Pump
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Breather
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Drain Plug
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Filter and Strainer
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Heater
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Heater Valve
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Level Float
    - Diesel - Emission Control - SCR (Selective Catalytic Reduction) - Urea/DEF Tank Temperature Control Valve & Sensor
    - Diesel - Engine Mounts
    - Diesel - Engine Valve Seats
    - Diesel - Engine Valve Springs
    - Diesel - Engine Valve Stem
    - Diesel - Engine Valve Train
    - Diesel - Exhaust Gas Recirculation (EGR) Cooler
    - Diesel - Exhaust Gas Recirculation (EGR) Valve
    - Diesel - Filter - Fuel (Pre-Fuel & Final)
    - Diesel - Filter - Inlet Air (Element or Cartridge Type)
    - Diesel - Filter - Lube Oil
    - Diesel - Fly Wheel
    - Diesel - Fuel
    - Diesel - Fuel Hoses
    - Diesel - Fuel Lines
    - Diesel - Fuel Tank
    - Diesel - Fuel Tank Breather or Vent
    - Diesel - Fuel Tank Strainer (if present)
    - Diesel - Fuel-Water Separator Element
    - Diesel - Gaskets, Seals, and O-rings (Internal and External Elastomer Type)
    - Diesel - Head Gasket
    - Diesel - High Pressure Fuel Pump
    - Diesel - Hydraulic Lifter (if present)
    - Diesel - Injector Tubing
    - Diesel - Injectors
    - Diesel - Linkages and Controls
    - Diesel - Lube Oil
    - Diesel - Lube Oil Pressure Control Device
    - Diesel - Lube Oil Pump
    - Diesel - Muffler
    - Diesel - PTO (present; not normally used in this application and is not expected to affect normal operation)
    - Diesel - Piston Wrist Pin Bearings
    - Diesel - Pistons
    - Diesel - Push Rods
    - Diesel - Radiator
    - Diesel - Radiator Cap
    - Diesel - Radiator Fan
    - Diesel - Radiator Hoses
    - Diesel - Radiator Tubing
    - Diesel - Starter
    - Diesel - Thermostat
    - Diesel - Timing Gears (if present)
    - Diesel - Turbocharger
    - Diesel - Turbocharger Exhaust Flex Hoses
    - Diesel - Turbocharger Exhaust Inlet Screen (if present)
    - Diesel - Valve Train - Rocker Arms with Rollers
    - Diesel - Vibration Damper or Harmonic Balancer
    - Diesel - Water Pump
    - Diesel - Wiring Harness
    - Diesel-Pump - Coupling - Elastomeric Element
    - Pump - Bearing Seals - Lip
    - Pump - Bearings - Rolling Element (Radial and Thrust)
    - Pump - Casing
    - Pump - Casing Drain or Stop Valve
    - Pump - Casing and Internals, if present
    - Pump - Check Valve - Disk Arm (if present)
    - Pump - Check Valve - Hinge Pin (if present)
    - Pump - Check Valve - Rubber Flapper (if present)
    - Pump - Check Valve - Seat Failure (Body or Disk)
    - Pump - Connections and Piping
    - Pump - Discharge and Suction Connections
    - Pump - Gaskets and O-Rings
    - Pump - Impeller and Wear Rings
    - Pump - Lubrication - Grease
    - Pump - Priming System
    - Pump - Seal - Mechanical Non-Oil Bath Type (process or seal water wetted)
    - Pump - Shaft
    - Skid - Diesel and Pump Base Plate or Frame
    - Trailer - Bed, Frame, and Lifting Lugs
    - Trailer - Electric Brakes
    - Trailer - Electric Lights
    - Trailer - Hitch or Coupling
    - Trailer - Hydraulic Brakes
    - Trailer - Levelers
    - Trailer - Suspension
    - Trailer - Tires
    - Trailer - Wheel Bearings
    - Trailer - Wheels Rims

    
    Goal: Generate candidate set for Bumper - Hydraulic.

    """
    initstep = True
    stateful = False
    asset_class = 'Test'

    experiment_name = "MyExperiment_" + asset_class + "_" + str(uuid.uuid4())
    experiment_id = mlflow.create_experiment(experiment_name)

    mdl = GenAIChatClient(
        name="ADesc",
        description="Asset Description",
        skill="Generate Asset Description",
        model=LLMsets[model_id],
        params=DEFAULT_CONFIG["params"],
        system_message=OriginalSysPrompt,
        credentials=DEFAULT_CONFIG["creds"],
        stateful=stateful,
        stream=True,
    )

    sentence = SystemPrompt + UserGoal

    ans = mdl.create(
        context=None,
        messages=[{"content": sentence, "role": "user"}],
        experiment_id=experiment_id,
    )
    
    return ans

print (p_ans)

p_ans = get_iso_asset_description(model_id=3)
print (p_ans)

"""
p_ans = get_asset_description(
    iteration=2,
    asset_class="Valve - Steam Turbine - Steam Stop Valve Balanced Type using a Single-Acting Actuator",
    iso=True,
    model_id=3,
)
print(p_ans)
"""

"""
p_ans = get_asset_description(iteration=2, asset_class="Electrical Substation Transformer")
print (p_ans)
"""
    

