import pandas as pd
from sentence_transformers import SentenceTransformer
import validation
import ast
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def find_majority_or_first_element(items):
    counts = Counter(items)
    majority_element, majority_count = counts.most_common(1)[0]
    if majority_count > len(items) / 2:
        return majority_element
    else:
        return items[0]


sets = [
    {
        "train": "failure_code.csv",
        "dest": "client_1_train_trial.csv",
        "index": "./all_mpnet_base_v2_val.csv",
        "cname": "dfsp_mpnet_base_v2_val",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_test_trial.csv",
        "index": "./all_mpnet_base_v2_test.csv",
        "cname": "dfsp_mpnet_base_v2_test",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_train_trial.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_supervised_val.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_supervised_val",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_test_trial.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_supervised_test.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_supervised_test",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_train_trial.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_unsup_simcse_val.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_unsup_simcse_val",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_test_trial.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_unsup_simcse_test.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_unsup_simcse_test",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_train_trial.csv",
        "index": "./e5_small_v2_val.csv",
        "cname": "e5_small_v2_val",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_test_trial.csv",
        "index": "./e5_small_v2_test.csv",
        "cname": "e5_small_v2_test",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_train_trial.csv",
        "index": "./e5_large_v2_val.csv",
        "cname": "e5_large_v2_val",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_test_trial.csv",
        "index": "./e5_large_v2_test.csv",
        "cname": "e5_large_v2_test",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_train_trial.csv",
        "index": "./slate_30m_english_rtrvr_val.csv",
        "cname": "slate_30m_english_rtrvr_val",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_test_trial.csv",
        "index": "./slate_30m_english_rtrvr_test.csv",
        "cname": "slate_30m_english_rtrvr_test",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_train_trial.csv",
        "index": "./slate_125m_english_rtrvr_val.csv",
        "cname": "slate_125m_english_rtrvr_val",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_test_trial.csv",
        "index": "./slate_125m_english_rtrvr_test.csv",
        "cname": "slate_125m_english_rtrvr_test",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_train_trial.csv",
        "index": "./bge_large_en_val.csv",
        "cname": "bge_large_en_val",
    },
    {
        "train": "failure_code.csv",
        "dest": "client_1_test_trial.csv",
        "index": "./bge_large_en_test.csv",
        "cname": "bge_large_en_test",
    },
]

totalks = [1, 3]

for k in totalks:

    for item in sets:
        df = pd.read_csv(item["dest"])
        df = df[["desc_and_uni_task", "problemcode"]]

        df.rename(columns={"label": "problemcode"}, inplace=True)
        df.rename(columns={"desc_and_uni_task": "short_descriptions"}, inplace=True)
        # df['gold_failure_locations'] = df['gold_failure_locations'].apply(lambda x: ast.literal_eval(x))

        df1 = pd.read_csv(item["index"])
        data = pd.read_csv(item["train"])
        data.rename(columns={"label": "gold_failure_locations"}, inplace=True)
        L = list(data["gold_failure_locations"])

        if k > 1:
            merged_column = df1[
                ["Top Index 1", "Top Index 2", "Top Index 3"]
            ].values.tolist()
            df1["merged_column"] = merged_column
            cand_col_name = item["cname"]
            df[cand_col_name] = df1["merged_column"].apply(lambda x: [L[xi] for xi in x])
            df[cand_col_name] = df1["merged_column"].apply(lambda x: [L[xi] for xi in x])
            df[cand_col_name] = df[cand_col_name].apply(
                lambda x: find_majority_or_first_element(x)
            )
        else:
            cand_col_name = item["cname"]
            df[cand_col_name] = df1["Top Index 1"].apply(lambda x: L[x])

        # print (df)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            df["problemcode"],
            df[cand_col_name],
            average="macro",
            zero_division=1,
        )
        print('macro', k, precision, recall, f1_score, item["cname"])

        # print (df)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            df["problemcode"],
            df[cand_col_name],
            average="micro",
            zero_division=1,
        )
        print('micro', k, precision, recall, f1_score, item["cname"])

        accuracy = accuracy_score(df["problemcode"], df[cand_col_name])
        print('accuracy', k, accuracy, 0, 0, item["cname"])