from typing import Callable, Dict, List, Optional, Tuple, Union

ssquestion = """
Here are some additional questions that a data scientist could ask a subject matter expert based on the provided conversation:

1. Can you provide more details about the types of sensors used to collect data on the wind turbine gearbox?
2. How is the data from the sensors transmitted and stored, and are there any potential issues with data loss or corruption?
3. How is the performance of the existing condition monitoring tools and techniques evaluated, and are there any metrics that could be used to measure the improvement with the anomaly detection model?
4. Are there any specific data quality issues or missing data that need to be addressed in the anomaly detection model?
5. How will the anomaly detection model be deployed and maintained in a distributed environment with multiple wind turbines?
6. Are there any specific communication protocols or interfaces that should be used for the integration of the anomaly detection model into the existing IoT/OT system?
7. Are there any specific data governance or data management policies that should be considered when building the anomaly detection model?
8. Are there any specific legal or regulatory requirements related to data privacy or data protection that should be taken into account when building the anomaly detection model?
9. Are there any specific data integration or data transformation requirements that should be considered when building the anomaly detection model?
10. Are there any specific data validation or data verification procedures that should be used to ensure the accuracy and reliability of the anomaly detection model?
11. Are there any specific data security or data access controls that should be implemented to protect the anomaly detection model and the associated data?
12. Are there any specific data lineage or data provenance requirements that should be considered when building the anomaly detection model?
13. Are there any specific data quality metrics or data quality standards that should be used to evaluate the performance of the anomaly detection model?
14. Are there any specific data visualization or reporting requirements for the output of the anomaly detection model, such as dashboards or alerts?
15. Are there any specific data security or data privacy requirements for the output of the anomaly detection model, such as data encryption or access controls?
16. Are there any specific data retention or data archiving requirements that should be considered when building the anomaly detection model?
17. Are there any specific data backup or data recovery procedures that should be implemented to ensure the availability and resilience of the anomaly detection model?
18. Are there any specific data archiving or data retention policies that should be considered when building the anomaly detection model?
19. Are there any specific data backup or data recovery procedures that should be implemented to ensure the availability and resilience of the anomaly detection model?
20. Are there any specific data archiving or data retention policies that should be considered when building the anomaly detection model?
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