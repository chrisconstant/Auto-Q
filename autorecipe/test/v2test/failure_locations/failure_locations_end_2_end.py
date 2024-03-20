import ray 

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

from generate_failure_locations_from_context_thinking_components import get_components
from generate_failure_locations_from_context_thinking_failure_mode import get_failure_modes
from generate_failure_locations_from_context_thinking_failure_location import get_failure_locations 
model_id = 3

def get_evaluation(model_id):
    components_str, component_list = get_components(sample_assetclass, sample_assetdesc, model_id=model_id)
    failuremode_ans = get_failure_modes(components_str, model_id=model_id)
    failure_locations = get_failure_locations(failuremode_ans, model_id=model_id)

    print (failure_locations)
import pandas as pd
import os

def get_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith("genai_context_docs"):
            csv_files.append(os.path.join(directory, filename))
    return csv_files

directory_path = "./experiment1_contextonly_withboundry/"
csv_files_list = get_csv_files(directory_path)

gold_df = pd.read_csv("./autoQ_val_data_input_for_experiments.csv")
asset_classes = list(gold_df["component_short_description"])

refs = []
for item_index, item in enumerate(asset_classes):
    item = item.replace("<", "-")
    item = item.replace("/", " ").replace(",", " ")
    item = item.replace(" ", "")
    notfound = True
    filename = ""
    for name in csv_files_list:
        if item in name:
            notfound = False
            filename = name
            break

    if notfound:
        pass
    else:
        refs.append(get_evaluation.remote(item, filename))

parallel_returns = ray.get(refs)

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v0-1-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
]

res = pd.DataFrame(parallel_returns)
model_initial = LLMsets[model_id].split("/")[1].split("-")[0]
directory_path = directory_path.replace('/','').replace('.','')
res.to_csv(f'guided_pipeline_generated_result_{model_initial}_{directory_path}.csv',index=False)

