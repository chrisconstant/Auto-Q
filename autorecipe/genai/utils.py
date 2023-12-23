from rouge_score import rouge_scorer


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

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # Sort the list by length
    sorted_list = sorted(input_list, key=len, reverse=True)

    # Initialize the return set
    result_list = []

    # Add the first element to the return set
    if sorted_list:
        result_list.append(sorted_list[0])

    # Iterate through the sorted list and add unique elements to the return set
    for element in sorted_list[1:]:
        is_duplicate = False
        for another_element in result_list:
            score = scorer.score(element, another_element)
            if score["rougeL"].fmeasure >= filter_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result_list.append(element)

    # Convert the set back to a list and return
    return result_list


def filter_and_sort_questions_using_reference(
    reference_list, input_list, filter_threshold=0.7, is_identical=False
):
    """_summary_

    :param input_list: _description_
    :type input_list: _type_
    :param filter_threshold: _description_, defaults to 0.7
    :type filter_threshold: float, optional
    :return: _description_
    :rtype: _type_
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # Initialize the return set
    result_list = []

    for element_index, element in enumerate(input_list):
        is_duplicate = False
        for another_element_index, another_element in enumerate(reference_list):
            if is_identical and element_index == another_element_index:
                continue
            score = scorer.score(element, another_element)
            if score["rougeL"].fmeasure >= filter_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result_list.append(element)

    # Convert the set back to a list and return
    return result_list
