import pandas as pd
from sentence_transformers import SentenceTransformer
import validation
import ast

sets = [
    {'train': "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv",
     'dest': "./data/all_ids_deduplicated_val_short_desc_to_failure_locations.csv",
     'mode': 'val'},
    {'train': "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv",
     'dest': "./data/all_ids_deduplicated_test_short_desc_to_failure_locations.csv",
     'mode': 'test'},
]

for item in sets:
    df = pd.read_csv("./data/all_ids_deduplicated_val_short_desc_to_failure_locations.csv")
    df.columns = ["TypeData.CompTypeID", "TypeData.GenCompType", "failure_locations"]

    df.rename(columns={"failure_locations": "gold_failure_locations"}, inplace=True)
    df.rename(columns={"TypeData.GenCompType": "short_descriptions"}, inplace=True)
    #df['gold_failure_locations'] = df['gold_failure_locations'].apply(lambda x: ast.literal_eval(x))

    df1 = pd.read_csv("./Mistral_sup_simcse_val.csv")
    # df2 = pd.read_csv('./Mistral_unsup_simcse_test.csv')
    df1 = pd.read_csv('./SentenceTransformer_sup_val.csv')

    data = pd.read_csv(
        "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv"
    )
    data.rename(columns={"failure_locations": "gold_failure_locations"}, inplace=True)
    L = list(data["gold_failure_locations"])

    df['index'] = range(df.shape[0])
    cand_col_name = "dfsp_Mistral_sup_simcse_test"
    df[cand_col_name] = df['index'].apply(lambda x: ast.literal_eval(L[x])) 

    print (df[cand_col_name])
    print (df['gold_failure_locations'])

    val_model = SentenceTransformer('all-mpnet-base-v2')
    step_1_data = validation.calculate_metrics(df, val_model, cand_col_name)

    table, plot = validation.get_precision_results(step_1_data)
    print(table)

    table, plot = validation.get_recall_results(step_1_data)
    print(table)

    prec_cols = [col for col in step_1_data if col.startswith('prec_')]
    cand_col_names = [col[5:] for col in prec_cols]
    f1_results = []
    for col in cand_col_names:
        f1 = validation.calculate_overall_f1(step_1_data,col)
        f1_results.append((f1, col))
    f1_results.sort(reverse=True)
    for res in f1_results:
        print("F1:", res[0], '\t', res[1])
