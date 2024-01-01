from rouge_score import rouge_scorer
import ray

@ray.remote
def check_is_element_duplicate(element, element1, filter_threshold, element1_index):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score = scorer.score(element, element1)
    if score["rougeL"].fmeasure >= filter_threshold:
        return (True, element1_index)
    return (False, element1_index)


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
