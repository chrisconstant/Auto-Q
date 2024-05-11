import pandas as pd
from sentence_transformers import SentenceTransformer
import validation
import ast

sets = [
    {
        "train": "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv",
        "dest": "./data/all_ids_deduplicated_val_short_desc_to_failure_locations.csv",
        "index": "./all_mpnet_base_v2_val.csv",
        "cname": "dfsp_mpnet_base_v2_val",
    },
    {
        "train": "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv",
        "dest": "./data/all_ids_deduplicated_test_short_desc_to_failure_locations.csv",
        "index": "./all_mpnet_base_v2_test.csv",
        "cname": "dfsp_mpnet_base_v2_test",
    },
    {
        "train": "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv",
        "dest": "./data/all_ids_deduplicated_val_short_desc_to_failure_locations.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_supervised_val.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_supervised_val",
    },
    {
        "train": "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv",
        "dest": "./data/all_ids_deduplicated_test_short_desc_to_failure_locations.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_supervised_test.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_supervised_test",
    },
    {
        "train": "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv",
        "dest": "./data/all_ids_deduplicated_val_short_desc_to_failure_locations.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_unsup_simcse_val.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_unsup_simcse_val",
    },
    {
        "train": "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv",
        "dest": "./data/all_ids_deduplicated_test_short_desc_to_failure_locations.csv",
        "index": "./Mistral_7B_Instruct_v2_mntp_unsup_simcse_test.csv",
        "cname": "Mistral_7B_Instruct_v2_mntp_unsup_simcse_test",
    },
]

totalks = [1, 3]

for k in totalks:

    for item in sets:
        df = pd.read_csv(item["dest"])
        df.columns = ["TypeData.CompTypeID", "TypeData.GenCompType", "failure_locations"]

        df.rename(columns={"failure_locations": "gold_failure_locations"}, inplace=True)
        df.rename(columns={"TypeData.GenCompType": "short_descriptions"}, inplace=True)
        # df['gold_failure_locations'] = df['gold_failure_locations'].apply(lambda x: ast.literal_eval(x))

        df1 = pd.read_csv(item["index"])
        data = pd.read_csv(item["train"])
        data.rename(columns={"failure_locations": "gold_failure_locations"}, inplace=True)
        L = list(data["gold_failure_locations"])

        if k > 1:
            merged_column = df1[["Top Index 1", "Top Index 2", "Top Index 3"]].values.tolist()
            df1["merged_column"] = merged_column
            cand_col_name = item["cname"]
            df[cand_col_name] = df1["merged_column"].apply(lambda x: [ast.literal_eval(L[xi]) for xi in x])
            df[cand_col_name] = df[cand_col_name].apply(lambda x: list(set([item for sublist in x for item in sublist])))
        else:
            cand_col_name = item["cname"]
            df[cand_col_name] = df1["Top Index 1"].apply(lambda x: L[x])

        print(df[cand_col_name])
        print(df["gold_failure_locations"])

        val_model = SentenceTransformer("all-mpnet-base-v2")
        step_1_data = validation.calculate_metrics(df, val_model, cand_col_name)

        table, plot = validation.get_precision_results(step_1_data)
        print(table)

        table, plot = validation.get_recall_results(step_1_data)
        print(table)

        prec_cols = [col for col in step_1_data if col.startswith("prec_")]
        cand_col_names = [col[5:] for col in prec_cols]
        f1_results = []
        for col in cand_col_names:
            f1 = validation.calculate_overall_f1(step_1_data, col)
            f1_results.append((f1, col))
        f1_results.sort(reverse=True)
        for res in f1_results:
            print("F1:", res[0], "\t", res[1])
