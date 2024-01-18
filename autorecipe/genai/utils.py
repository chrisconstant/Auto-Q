from rouge_score import rouge_scorer
import ray
import numpy as np
import nltk as nlp
import matplotlib.pyplot as plt
import re
from nltk.probability import FreqDist
import math
from collections import Counter
from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams

@ray.remote
def check_is_element_duplicate(element, element1, filter_threshold, element1_index):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score = scorer.score(element, element1)
    if score["rougeL"].fmeasure >= filter_threshold:
        return (True, element1_index)
    return (False, element1_index)

@ray.remote
def get_TTR(questions):
    # List to store the TTR values
    TTR_List = []
    # List to store the ID of the article
    Article_Number = []
    for counter in range(len(questions)):
        document = questions[counter]
        # Remove all special characters using this regex
        document = re.sub(r"[^\w]", " ", document)
        # Convert Document to Lower Case
        document = document.lower()

        tokens = nlp.word_tokenize(document)
        total_tokens = len(tokens)
        freq_dist = FreqDist(tokens)
        num_types = len(freq_dist)
        normalized_ttr = num_types / math.sqrt(total_tokens)

        TTR = normalized_ttr
        # Append TTR to TTR_List
        TTR_List.append(TTR)
        # Append counter to Article_Number
        Article_Number.append(counter)

    return np.mean(TTR_List)

@ray.remote
def call_QuestionUserfulClassifier(sentence):
    api_key = "pak-GeJBIH1iVY5FuvvSjsc-BrQI_iOdFoJLLQXOzJ3zRuQ"
    api_url = "https://bam-api.res.ibm.com"
    creds = Credentials(api_key, api_endpoint=api_url)

    print("\n------------- Example (Model Talk)-------------\n")

    bob_params = GenerateParams(decoding_method="greedy", max_new_tokens=25, temperature=1)
    QuestionUserfulClassifier = Model(
        "flan-t5-xl-pt-VQ5QsUX4-2024-01-02-08-18-33",
        params=bob_params,
        credentials=creds,
    )

    q_response = QuestionUserfulClassifier.generate([sentence])
    q_gen = q_response[0].generated_text
    if '0' in q_gen:
        return 0
    return 1        

def filter_questions_using_TTR(questions, ttr_threshold=2):
    """_summary_

    :param questions: _description_
    :type questions: _type_
    :param ttr_threshold: _description_, defaults to 2
    :type ttr_threshold: int, optional
    :return: _description_
    :rtype: _type_
    """
    screen_questions = []
    refs = []
    for item in questions:
        refs.append(get_TTR.remote([item]))
    refs_responses = ray.get(refs)
    for index, item in enumerate(questions):
        if refs_responses[index] >= ttr_threshold:
            screen_questions.append(item)
    return screen_questions

def filter_questions_using_Flan(questions):
    """_summary_

    :param questions: _description_
    :type questions: _type_
    :param ttr_threshold: _description_, defaults to 2
    :type ttr_threshold: int, optional
    :return: _description_
    :rtype: _type_
    """
    screen_questions = []
    refs = []
    for item in questions:
        refs.append(call_QuestionUserfulClassifier.remote(item))
    refs_responses = ray.get(refs)
    for index, item in enumerate(questions):
        if refs_responses[index] == 1:
            screen_questions.append(item)
    return screen_questions


def filter_and_sort_questions(input_list, filter_threshold=0.7):
    """_summary_

    :param input_list: _description_
    :type input_list: _type_
    :param filter_threshold: _description_, defaults to 0.7
    :type filter_threshold: float, optional
    :return: _description_
    :rtype: _type_
    """
    if len(input_list) <= 1:
        return input_list

    # Sort the list by length
    sorted_list = sorted(input_list, key=len, reverse=True)

    # Initialize the return set
    result_list = []

    # duplicate index
    duplicate_id = []

    # Iterate through the sorted list and add unique elements to the return set
    for element_index, element in enumerate(sorted_list):

        if element_index in duplicate_id:
            continue

        # select the results
        result_list.append(element)

        # remove duplicate
        refs = []
        for element1_index, element1 in enumerate(sorted_list[element_index:]):
            refs.append(
                check_is_element_duplicate.remote(
                    element, element1, filter_threshold, element_index + element1_index
                )
            )

        refs_responses = ray.get(refs)
        for item in refs_responses:
            if item[0]:
                duplicate_id.append(item[1])

    # Convert the set back to a list and return
    return result_list

@ray.remote
def check_is_duplicate(element, reference_list, filter_threshold):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    is_duplicate = False
    for _, another_element in enumerate(reference_list):
        score = scorer.score(element, another_element)
        if score["rougeL"].fmeasure >= filter_threshold:
            is_duplicate = True
            break
    return is_duplicate


def filter_questions_using_reference(reference_list, input_list, filter_threshold=0.7):
    """_summary_

    :param input_list: _description_
    :type input_list: _type_
    :param filter_threshold: _description_, defaults to 0.7
    :type filter_threshold: float, optional
    :return: _description_
    :rtype: _type_
    """
    # Initialize the return set
    result_list = []
    refs = []

    for _, element in enumerate(input_list):
        refs.append(check_is_duplicate.remote(element, reference_list, filter_threshold))

    refs_responses = ray.get(refs)

    for index, element in enumerate(input_list):
        if not refs_responses[index]:
            result_list.append(element)

    # Convert the set back to a list and return
    return result_list
