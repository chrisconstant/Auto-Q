import pandas as pd
from sentence_transformers import SentenceTransformer
import validation
import ast
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

def find_majority_or_first_element(items):
    counts = Counter(items)
    majority_element, majority_count = counts.most_common(1)[0]
    if majority_count > len(items) / 2:
        return majority_element
    else:
        return items[0]

sets = [
    {
        "train": "./train.csv",
        "dest": "./dev.csv",
        "index": "./all_mpnet_base_v2_val.csv",
        "cname": "dfsp_mpnet_base_v2_val",
    },
    {
        "train": "./train.csv",
        "dest": "./test.csv",
        "index": "./all_mpnet_base_v2_test.csv",
        "cname": "dfsp_mpnet_base_v2_test",
    },
    {
        "train": "./train.csv",
        "dest": "./dev.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_supervised_val.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_supervised_val",
    },
    {
        "train": "./train.csv",
        "dest": "./test.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_supervised_test.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_supervised_test",
    },
    {
        "train": "./train.csv",
        "dest": "./dev.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_unsup_simcse_val.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_unsup_simcse_val",
    },
    {
        "train": "./train.csv",
        "dest": "./test.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_unsup_simcse_test.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_unsup_simcse_test",
    },
]

k = 1

for item in sets:
    df = pd.read_csv(item["dest"])
    df.columns = ["workordertext","description","label"]

    df.rename(columns={"label": "gold_failure_locations"}, inplace=True)
    df.rename(columns={"workordertext": "short_descriptions"}, inplace=True)
    # df['gold_failure_locations'] = df['gold_failure_locations'].apply(lambda x: ast.literal_eval(x))

    df1 = pd.read_csv(item["index"])
    data = pd.read_csv(item["train"])
    data.rename(columns={"label": "gold_failure_locations"}, inplace=True)
    L = list(data["gold_failure_locations"])


    if k > 1:
        merged_column = df1[["Top Index 1", "Top Index 2", "Top Index 3"]].values.tolist()
        df1["merged_column"] = merged_column
        cand_col_name = item["cname"]
        df[cand_col_name] = df1["merged_column"].apply(lambda x: [L[xi] for xi in x])
        df[cand_col_name] = df1["merged_column"].apply(lambda x: [L[xi] for xi in x])
        df[cand_col_name] = df[cand_col_name].apply(lambda x: find_majority_or_first_element(x))
    else:
        cand_col_name = item["cname"]
        df[cand_col_name] = df1["Top Index 1"].apply(lambda x: L[x])

    precision, recall, f1_score, _ = precision_recall_fscore_support(df['gold_failure_locations'], df[cand_col_name], average='weighted', zero_division=1)
    print (k, precision, recall, f1_score, item['cname'])
