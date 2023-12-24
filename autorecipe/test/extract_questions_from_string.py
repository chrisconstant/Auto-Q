from typing import Callable, Dict, List, Optional, Tuple, Union

squestrion = """
1. Can you provide more details about the vibration sensors used in wind turbine gearboxes? How do they work, and what are the common failure modes?
2. How do temperature sensors work in wind turbine gearboxes? What are the common failure modes, and how do they affect the operation of the gearbox?
3. Can you discuss the pressure sensors used in wind turbine gearboxes? How do they work, and what are the common failure modes?
4. How do speed sensors work in wind turbine gearboxes? What are the common failure modes, and how do they affect the operation of the gearbox?
5. Can you provide more details about the torque sensors used in wind turbine gearboxes? How do they work, and what are the common failure modes?
6. How do acoustic sensors work in wind turbine gearboxes? What are the common failure modes, and how do they affect the operation of the gearbox?
7. Can you discuss the oil sensors used in wind turbine gearboxes? How do they work, and what are the common failure modes?
8. How do control system sensors work in wind turbine gearboxes? What are the common failure modes, and how do they affect the operation of the gearbox?
9. Can you provide examples of how the data collected from these sensors can be used to detect anomalies in the wind turbine gearbox?
10. How often should the sensors in the wind turbine gearbox be calibrated, and what are the potential consequences of improper calibration?
11. Can you discuss the potential consequences of sensor failure in a wind turbine gearbox? How does it affect the operation of the wind turbine, and what are the potential costs associated with repairs and maintenance?
12. How do the sensors in the wind turbine gearbox communicate with the control system, and what are the potential failure modes in the communication system?
13. Can you provide examples of how the data collected from the sensors in the wind turbine gearbox can be used to predict potential failures before they occur?
14. How does the control system in the wind turbine gearbox use the data collected from the sensors to make decisions, and what are the potential failure modes in the decision-making process?
15. Can you discuss the potential benefits of using machine learning algorithms to analyze the data collected from the sensors in the wind turbine gearbox? How can they improve the accuracy of anomaly detection and prediction of failures?
"""

ssquestion = """
  Sure, I can help you with that. To build an anomaly model using real-time time series sensor data obtained from IoT/OT systems for wind turbine gearbox, I would need to ask the following questions to subject matter experts in a sequential manner:\n\n1. Can you provide an overview of the wind turbine gearbox and its components? This will help me understand the important components for which the anomaly model should be built.\n2. What are the most common failure modes for wind turbine gearboxes? This will help me identify the failure modes that the anomaly model should be able to detect.\n3. Which sensors are available for collecting data from the wind turbine gearbox? This will help me understand the type of data that can be used to build the anomaly model.\n4. What is the sampling rate for the sensor data? This will help me determine the frequency of data collection and the granularity of the data.\n5. Are there any specific failure modes that are more critical than others? This will help me prioritize the failure modes that the anomaly model should detect.\n6. Are there any specific components that are more prone to failure than others? This will help me focus the anomaly model on the most critical components.\n7. How does the wind turbine gearbox operate, and what are the normal operating conditions? This will help me understand the normal behavior of the gearbox and detect anomalies.\n8. Are there any external factors that can affect the operation of the wind turbine gearbox, such as temperature, humidity, or vibration? This will help me understand the potential sources of noise in the data.\n9. Are there any existing maintenance schedules or procedures that can be leveraged to inform the anomaly model? This will help me understand the current maintenance practices and how the anomaly model can be integrated into them.\n10. Are there any specific regulatory or compliance requirements that the anomaly model must adhere to? This will help me ensure that the anomaly model meets the necessary standards and regulations.\n11. Are there any additional data sources or information that can be leveraged to improve the accuracy of the anomaly model, such as historical data or expert knowledge? This will help me identify potential sources of additional information that can be used to improve the model.\n12. What is the desired level of accuracy for the anomaly model, and what are the consequences of false positives or false negatives? This will help me understand the performance requirements for the model and the potential impact of errors.\n\nBy asking these questions, I can gather the necessary information to build an effective anomaly model using real-time time series sensor data obtained from IoT/OT systems for wind turbine gearbox.
"""

def content_str(content: Union[str, List]) -> str:
    if type(content) is str:
        return content
    rst = ""
    for item in content:
        if item["type"] == "text":
            rst += item["text"]
        else:
            assert (
                isinstance(item, dict) and item["type"] == "image_url"
            ), "Wrong content format."
            rst += "<image>"
    return rst


def extract_fun(text):
    chat_agent_response = content_str(text)
    questions_start_index = chat_agent_response.find("1. ")

    if "\n\n" in chat_agent_response:
        questions_end_index = chat_agent_response.rfind("\n\n") + 2
    else:
        questions_end_index = len(chat_agent_response)


    if questions_end_index == questions_start_index:
        questions_end_index = len(chat_agent_response)

    print (questions_start_index, questions_end_index)
    questions_string = chat_agent_response[
        questions_start_index:questions_end_index
    ]
    print (questions_start_index)

    # Splitting the questions into a list
    questions_list = questions_string.split("\n")

    # Removing empty elements from the list
    questions_list = [
        question.strip() for question in questions_list if question.strip()
    ]
    final_questions = []
    for item in questions_list:
        first_space_index = item.find(" ")
        if first_space_index != -1:
            final_questions.append(item[first_space_index + 1 :])
        else:
            final_questions.append(item)

    # Printing the list of questions
    return final_questions


ans = extract_fun(ssquestion)
print (ans)