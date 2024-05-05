import pandas as pd
import torch
from llm2vec import LLM2Vec
from sentence_transformers import SentenceTransformer

sets = [
    {'train': "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv",
     'dest': "./data/all_ids_deduplicated_val_short_desc_to_failure_locations.csv",
     'mode': 'val',
     },
    {'train': "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv",
     'dest': "./data/all_ids_deduplicated_test_short_desc_to_failure_locations.csv",
     'mode': 'test',
     },
]

models = [
    {'mdl': "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
     'peft_model_name_or_path': "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
     'mode': 'Mistral_7B_Instruct_v2_mntp_unsup_simcse',
     'type': 'LLM2vec'
     },
    {'mdl': "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
     'peft_model_name_or_path': "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
     'mode': 'Mistral_7B_Instruct_v2_mntp_supervised',
     'type': 'LLM2vec'
     },
    {'mdl': "all-mpnet-base-v2",
     'mode': 'all_mpnet_base_v2',
     'type': 'ST'
     },
]

for db in sets:
    for mdl in models:
        if mdl['type'] == 'LLM2vec':
            l2v = LLM2Vec.from_pretrained(
                mdl['mdl'],
                peft_model_name_or_path=mdl['peft_model_name_or_path'],
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )

            prefixmatch = mdl['mode'] + '_' + db['mode']

            df = pd.read_csv(db['train'])
            documents = list(df["TypeData.GenCompType"])
            d_reps = l2v.encode(documents)

            df = pd.read_csv(db['dest'])
            querys = list(df["TypeData.GenCompType"])
            instruction = "Given an industrial equipment, retrieve similar equipment:"
            queries = [instruction + " " + item for item in querys]
            q_reps = l2v.encode(queries)

            # passing the results
            q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
            d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
            cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

            top_values, top_indices = torch.topk(cos_sim, k=3, dim=1)
            top_values_list = top_values.tolist()
            top_indices_list = top_indices.tolist()

            # Convert lists to pandas DataFrames
            df_values = pd.DataFrame(
                top_values_list, columns=["Top Value 1", "Top Value 2", "Top Value 3"]
            )
            df_indices = pd.DataFrame(
                top_indices_list, columns=["Top Index 1", "Top Index 2", "Top Index 3"]
            )

            # Concatenate DataFrames horizontally (along columns)
            df_merged = pd.concat([df_values, df_indices], axis=1)
            df_merged.to_csv(prefixmatch + ".csv", index=False)
        elif db['type'] == 'ST':
            sentence_model = SentenceTransformer(mdl['mdl'])
            prefixmatch = mdl['mode'] + '_' + db['mode']

            df = pd.read_csv(db['train'])
            documents = list(df["TypeData.GenCompType"])

            df = pd.read_csv(db['dest'])
            querys = list(df["TypeData.GenCompType"])

            train_reps = sentence_model.encode(documents)
            test_reps = sentence_model.encode(querys)

            # Normalize representations
            train_reps_norm = torch.nn.functional.normalize(torch.tensor(train_reps), p=2, dim=1)
            test_reps_norm = torch.nn.functional.normalize(torch.tensor(test_reps), p=2, dim=1)

            # Calculate cosine similarity
            train_cos_sim = torch.mm(train_reps_norm, train_reps_norm.transpose(0, 1))
            test_cos_sim = torch.mm(test_reps_norm, train_reps_norm.transpose(0, 1))

            # Get top k indices and values
            def process_cos_sim(cos_sim, prefixmatch):
                top_values, top_indices = torch.topk(cos_sim, k=3, dim=1)
                top_values_list = top_values.tolist()
                top_indices_list = top_indices.tolist()

                # Convert lists to pandas DataFrames
                df_values = pd.DataFrame(
                    top_values_list, columns=[f"Top Value {i+1}" for i in range(3)]
                )
                df_indices = pd.DataFrame(
                    top_indices_list, columns=[f"Top Index {i+1}" for i in range(3)]
                )

                # Concatenate DataFrames horizontally (along columns)
                df_merged = pd.concat([df_values, df_indices], axis=1)
                df_merged.to_csv(prefixmatch + ".csv", index=False)

