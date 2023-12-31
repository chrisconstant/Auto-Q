from typing import Callable, Dict, List, Optional, Tuple, Union

ssquestion = """
  Sure, here are some additional questions that could be asked to a subject matter expert to gather more information about the wind turbine gearbox and its potential failure modes:

1. Can you provide more information about the specific components of the gearbox that are most critical to its operation and potential failure modes?
2. How do the failure modes of the gearbox components relate to each other? For example, if a bearing fails, is it likely to lead to failure of other components?
3. What are the most common causes of gearbox failure in wind turbines, and how do they relate to the operating conditions of the turbine?
4. Can you provide examples of how the degradation mechanisms of the gearbox components can lead to failure, and how these failures can be detected through vibration analysis, oil analysis, or visual inspection?
5. What are the typical warning signs or indicators that a gearbox failure is imminent, and how can these be detected in real-time?
6. Are there any specific maintenance tasks or procedures that can help prevent gearbox failures, and how often should these tasks be performed?
7. How do environmental factors such as temperature, humidity, and vibration affect the operation and lifespan of the gearbox components?
8. Are there any specific sensor data or other metrics that can be used to monitor the health of the gearbox in real-time, and how can these data be used to build an anomaly model?
9. Can you provide examples of how a gearbox failure can impact the overall performance and reliability of the wind turbine, and what are the potential consequences of a failure?
10. Are there any industry standards or best practices for gearbox maintenance and failure prevention in wind turbines, and how can these be incorporated into the anomaly model?
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
    print (questions_start_index)
    if '1. ' not in chat_agent_response:
        if '* ' in chat_agent_response:
            questions_start_index = chat_agent_response.find("* ")
    print (questions_start_index)

    if "\n\n" in chat_agent_response:
        questions_end_index = chat_agent_response.rfind("\n\n") + 2
    else:
        questions_end_index = len(chat_agent_response)

    if questions_end_index == questions_start_index:
        if "\n" in chat_agent_response:
            questions_end_index = chat_agent_response.rfind("\n") + 2
        else:
            questions_end_index = len(chat_agent_response)

    questions_string = chat_agent_response[
        questions_start_index:questions_end_index
    ]

    print (questions_string)
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