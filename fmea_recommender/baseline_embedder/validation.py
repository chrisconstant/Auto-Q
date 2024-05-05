import math
import re

import torch
from sentence_transformers import util
from rouge_score import rouge_scorer

def list_comparison_scores(list1, list2, validation_model):
    """Input: Two lists of strings.
    Output: For each string in 'list1', the top-k matches and scores in 'list2'"""
    list1_embeddings = validation_model.encode(list1, convert_to_tensor=True)
    list2_embeddings = validation_model.encode(list2, convert_to_tensor=True)
    comparison_results = util.dot_score(list1_embeddings, list2_embeddings)
    if len(list1) > 1 and len(list2) > 1:
        k = 2
    else:
        k = 1
    top_k_matches = torch.topk(comparison_results, k, dim=1).indices.tolist()
    top_k_scores = torch.topk(comparison_results, k, dim=1).values.tolist()
    return top_k_matches, top_k_scores


def get_missing_failure_locations(
    cand_list, gold_list, validation_model, threshold=0.7
):
    """The purpose of this method is to compare lists of things, either lists of 'failure locations' or
    lists of 'degradation mechanisms'.
    Input:
        cand_list - generated list of 'failure locations'/'degradation mechanisms';
        gold_list - gold list of 'failure locations'/'degradation mechanisms';
        validation_model - a text encoding model from the sentence_transformers package;
        threshold - similarity measures above this threshold value are considered to be matches.
    Output:
        missing_list - elements in gold_list, but not in cand_list
    """
    if len(cand_list) == 0:
        return list(set(gold_list))
    top_k_matches, top_k_scores = list_comparison_scores(
        cand_list, gold_list, validation_model
    )
    already_matched_gold = set()
    for cnt, _ in enumerate(cand_list):
        for rank, gold_idx in enumerate(top_k_matches[cnt]):
            gold_match = gold_list[gold_idx]
            if gold_match not in already_matched_gold:
                if top_k_scores[cnt][rank] > threshold:
                    already_matched_gold.add(gold_match)
                break
    missing_egs = set(gold_list) - already_matched_gold

    return list(missing_egs)


def get_common_failure_locations(cand_list, gold_list, validation_model, threshold=0.7):
    """The purpose of this method is to compare lists of things, either lists of 'failure locations' or
    lists of 'degradation mechanisms'.
    Input:
        cand_list - generated list of 'failure locations'/'degradation mechanisms';
        gold_list - gold list of 'failure locations'/'degradation mechanisms';
        validation_model - a text encoding model from the sentence_transformers package;
        threshold - similarity measures above this threshold value are considered to be matches.
    Output:
        common_list - elements common to both lists
    """
    if len(cand_list) == 0:
        return []
    top_k_matches, top_k_scores = list_comparison_scores(
        cand_list, gold_list, validation_model
    )
    common_list = list()
    for cnt, _ in enumerate(cand_list):
        for rank, gold_idx in enumerate(top_k_matches[cnt]):
            gold_match = gold_list[gold_idx]
            if gold_match not in common_list:
                if top_k_scores[cnt][rank] > threshold:
                    common_list.append(gold_match)
                break

    return list(common_list)


def list_comparison_scores_karol(list1, list2, validation_model):
    """Input: Two lists of strings.
    Output: For each string in 'list1', the top-k matches and scores in 'list2'"""
    list1_embeddings = validation_model.encode(list1, convert_to_tensor=True)
    list2_embeddings = validation_model.encode(list2, convert_to_tensor=True)
    comparison_results = util.dot_score(list1_embeddings, list2_embeddings)
    k = len(list2)
    top_k_matches = torch.topk(comparison_results, k, dim=1).indices.tolist()
    top_k_scores = torch.topk(comparison_results, k, dim=1).values.tolist()
    return top_k_matches, top_k_scores


def get_fuzzy_matching_items(a, reference_list, validation_model, threshold=0.7):
    """Get all items in the reference list that fuzzy match a.
    Input:
        a - item to be matched
        reference_list - list of items to be matched against;
        validation_model - a text encoding model from the sentence_transformers package;
        threshold - similarity measures above this threshold value are considered to be matches.
    Output:
        common_list - elements common to both lists
    """
    top_k_matches, top_k_scores = list_comparison_scores_karol(
        [a], reference_list, validation_model
    )
    common_list = set()
    for rank, gold_idx in enumerate(top_k_matches[0]):
        gold_match = reference_list[gold_idx]
        if gold_match not in common_list:
            if top_k_scores[0][rank] > threshold:
                common_list.add(gold_match)

    return list(common_list)


def calculate_precision(cand_list, gold_list, validation_model, threshold=0.7):
    """The purpose of this method is to compare lists of things, either lists of 'failure locations' or
    lists of 'degradation mechanisms'.
    Input:
        cand_list - generated list of 'failure locations'/'degradation mechanisms';
        gold_list - gold list of 'failure locations'/'degradation mechanisms';
        validation_model - a text encoding model from the sentence_transformers package;
        threshold - similarity measures above this threshold value are considered to be matches.
    Output:
        precision - the fraction of things from the cand_list that are contained in the gold_list
    """
    # Edge cases.
    if len(cand_list) == 0: # FP = 0, so precision is 1.
        return 1.0
    elif len(cand_list) > 0 and len(gold_list) == 0:
        return 0.0

    top_k_matches, top_k_scores = list_comparison_scores(
        cand_list, gold_list, validation_model
    )

    already_matched_gold = set()
    for cnt, _ in enumerate(cand_list):
        for rank, gold_idx in enumerate(top_k_matches[cnt]):
            gold_match = gold_list[gold_idx]
            if gold_match not in already_matched_gold:
                if top_k_scores[cnt][rank] > threshold:
                    already_matched_gold.add(gold_match)
                break

    precision = len(already_matched_gold) / len(set(cand_list))
    return precision


def calculate_recall(cand_list, gold_list, validation_model, threshold=0.7):
    """The purpose of this method is to compare lists of things, either lists of 'failure locations' or
    lists of 'degradation mechanisms'.
    Input:
        cand_list - generated list of 'failure locations'/'degradation mechanisms';
        gold_list - gold list of 'failure locations'/'degradation mechanisms';
        validation_model - a text encoding model from the sentence_transformers package;
        threshold - similarity measures above this threshold value are considered to be matches.
    Output:
        recall - the fraction of things from the gold_list that are contained in the cand_list
    """

    # Edge Cases
    if len(gold_list) == 0:     # No TPs (e.g., no components)
        # We could return 1.0 here, but since we are more interested overall recall across all examples
        # its best to simply not consider these cases when calculating average recall. 
        # Pandas aggregation functions such as mean ignore NAN values.
        return math.nan
    elif len(cand_list) == 0 and len(gold_list) > 0:
        # We found no TPs (e.g., no components)
        return 0.0

    top_k_matches, top_k_scores = list_comparison_scores(
        gold_list, cand_list, validation_model
    )
    already_matched_cand = set()
    for cnt, _ in enumerate(gold_list):
        for rank, cand_idx in enumerate(top_k_matches[cnt]):
            cand_match = cand_list[cand_idx]
            if cand_match not in already_matched_cand:
                if top_k_scores[cnt][rank] > threshold:
                    already_matched_cand.add(cand_match)
                break

    recall = len(already_matched_cand) / len(set(gold_list))
    return recall

def extract_things_from_string(text):
    fls = re.findall("(\{|\[)(.*?)(\}|\])", text)
    fls = fls[0][1].split("', '")
    fls = [fl.replace("'", "") for fl in fls]

    return fls



def parse_failure_locations(row, gold_column="gold_failure_locations"):
    gold_vals = extract_things_from_string(row[gold_column])
    return gold_vals


def calculate_precision_wrapper(
    row, val_model, candidate_column, gold_column="gold_failure_locations"
):
    if isinstance(row[candidate_column], str):
        cand_vals = extract_things_from_string(row[candidate_column])
    else:
        cand_vals = row[candidate_column]

    if isinstance(row[gold_column], str):
        gold_vals = extract_things_from_string(row[gold_column])
    else:
        gold_vals = row[gold_column]

    prec = calculate_precision(
        cand_list=cand_vals,
        gold_list=gold_vals,
        validation_model=val_model,
        threshold=0.7,
    )

    return prec


def calculate_f1(prec, recall):
    f1 = 2 * (prec * recall) / (prec + recall + 1.0e-30)
    # Add a small float in case precision and recall are both 0.
    return f1


def calculate_f1_wrapper(row, precision_column, recall_column):
    return calculate_f1(row[precision_column], row[recall_column])

def calculate_recall_wrapper(
    row, val_model, candidate_column, gold_column="gold_failure_locations"
):
    if isinstance(row[candidate_column], str):
        cand_vals = extract_things_from_string(row[candidate_column])
    else:
        cand_vals = row[candidate_column]

    if isinstance(row[gold_column], str):
        gold_vals = extract_things_from_string(row[gold_column])
    else:
        gold_vals = row[gold_column]

    recall = calculate_recall(
        cand_list=cand_vals,
        gold_list=gold_vals,
        validation_model=val_model,
        threshold=0.7,
    )

    return recall


def get_common_failure_locations_wrapper(row, val_model, candidate_column, gold_column):
    cand_vals = row[candidate_column]
    gold_vals = extract_things_from_string(row[gold_column])
    common_list = get_common_failure_locations(
        cand_list=cand_vals,
        gold_list=gold_vals,
        validation_model=val_model,
        threshold=0.7,
    )

    return common_list


def get_missing_failure_locations_wrapper(
    row, val_model, candidate_column, gold_column
):
    cand_vals = row[candidate_column]
    gold_vals = extract_things_from_string(row[gold_column])
    missing_list = get_missing_failure_locations(
        cand_list=cand_vals,
        gold_list=gold_vals,
        validation_model=val_model,
        threshold=0.7,
    )

    return missing_list


def process_single_shot(single_shot):
    single_shot = single_shot.replace("{", "").replace("}", "").split("', '")
    single_shot = [fl.replace("'", "") for fl in single_shot]
    single_shot = [fl.rstrip(",") for fl in single_shot]
    single_shot
    return set(single_shot)


def merge_multiple_shots(few_shot_fls):
    merged_fls = set()
    for eg in few_shot_fls:
        merged_fls = merged_fls.union(process_single_shot(eg))
    return list(merged_fls)


def get_cleaned_few_shots(all_few_shot_fls):
    all_few_shot_fls_cleaned = []
    for _, few_shot_fls in enumerate(all_few_shot_fls):
        all_few_shot_fls_cleaned.append(merge_multiple_shots(few_shot_fls))
    return all_few_shot_fls_cleaned

def calculate_metrics(df, val_model, candidate_col):
    """Calculate Precision, Recall, F1 and Rouge*"""
    df["prec_" + candidate_col] = df.apply(
        lambda row: calculate_precision_wrapper(
            row,
            val_model,
            candidate_column=candidate_col,
            gold_column="gold_failure_locations",
        ),
        axis=1,
    )
    df["rec_" + candidate_col] = df.apply(
        lambda row: calculate_recall_wrapper(
            row,
            val_model,
            candidate_column=candidate_col,
            gold_column="gold_failure_locations",
        ),
        axis=1,
    )

    df["f1_" + candidate_col] = df.apply(
        lambda row: calculate_f1_wrapper(
            row,
            precision_column="prec_" + candidate_col,
            recall_column="rec_" + candidate_col,
        ),
        axis=1,
    )

    for rm in ["rouge1", "rouge2", 'rougeL', 'rougeLsum']:
        df[rm + "_" + candidate_col] =\
        df.apply(lambda row: calculate_rouge_score_wrapper_lists(\
                                            row,
                                            candidate_column = candidate_col,
                                            gold_column='gold_failure_locations',
                                            rouge_metric=rm), axis=1)
    return df


# See HERE: https://pypi.org/project/rouge-score/
# ROUGE-N (N-gram) scoring
# ROUGE-L (Longest Common Subsequence) scoring
def get_rouge_score(prediction, reference, rouge_metric):
    """Calculate rouge score"""
    scorer = rouge_scorer.RougeScorer([rouge_metric])
    metrics = scorer.score(prediction, reference)
    rouge_f1_score = metrics[rouge_metric][-1]
    return rouge_f1_score

def calculate_rouge_score_wrapper(row,
                                  candidate_column,
                                  gold_column,
                                  rouge_metric="rouge1"):
    """The rouge metrics should be one of ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']"""
    candidate_str = row[candidate_column]
    gold_str = row[gold_column]
    rouge_f1_score = get_rouge_score(prediction = candidate_str,
                            reference = gold_str,
                            rouge_metric=rouge_metric)

    return rouge_f1_score

def calculate_rouge_score_wrapper_lists(row,
                                  candidate_column,
                                  gold_column,
                                  rouge_metric="rouge1"):
    """The rouge metrics should be one of ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'].
    cand_vals - a list of candidate values.
    gold_column - a string representation of a list"""
    cand_vals = row[candidate_column]
    gold_vals = extract_things_from_string(row[gold_column])
    # Sort lists (in-place)
    cand_vals.sort()
    gold_vals.sort()

    rouge_f1_score = get_rouge_score(prediction = ' '.join(cand_vals),
                            reference = ' '.join(gold_vals),
                            rouge_metric=rouge_metric)

    return rouge_f1_score


def get_precision_results(df):
    """Get a table and chart of the mean precision of each model."""
    summary = df.describe()
    summary_precision = summary[summary.columns[summary.columns.str.startswith('prec')]]
    table = summary_precision.iloc[1:].sort_values(by='mean', axis='columns', ascending=False)
    ax = summary_precision.loc[['mean', 'min', '25%', '50%', '75%', 'max']].plot(kind='line', title="PRECISION")
    plot = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.4))
    
    return table, plot

def get_recall_results(df):
    """Get the mean recall of each model."""
    summary = df.describe()
    summary_recall = summary[summary.columns[summary.columns.str.startswith('rec')]]
    table = summary_recall.iloc[1:].sort_values(by='mean', axis='columns', ascending=False)
    ax = summary_recall.loc[['mean', 'min', '25%', '50%', '75%', 'max']].plot(kind='line', title="RECALL")
    plot = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.4))
    return table, plot

def get_f1_results(df):
    """Get the mean F1 of each model."""
    summary = df.describe()
    summary_recall = summary[summary.columns[summary.columns.str.startswith('f1')]]
    table = summary_recall.iloc[1:].sort_values(by='mean', axis='columns', ascending=False)
    ax = summary_recall.loc[['mean', 'min', '25%', '50%', '75%', 'max']].plot(kind='line', title="F1")
    plot = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.4))
    return table, plot


def calculate_overall_f1(df, cand_col_name):
    prec = df['prec_'+ cand_col_name ].mean()
    rec = df['rec_'+ cand_col_name ].mean()

    f1 = calculate_f1(prec, rec)
    return f1
