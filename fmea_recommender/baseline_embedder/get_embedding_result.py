import pandas as pd
import torch
from llm2vec import LLM2Vec

l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

prefixmatch = 'MetaLlama3_sup'

df = pd.read_csv(
    "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv"
)
documents = list(df["TypeData.GenCompType"])
d_reps = l2v.encode(documents)

df = pd.read_csv("./data/all_ids_deduplicated_test_short_desc_to_failure_locations.csv")
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
df_merged.to_csv(prefixmatch + "_test.csv", index=False)

df = pd.read_csv("./data/all_ids_deduplicated_val_short_desc_to_failure_locations.csv")
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
df_merged.to_csv(prefixmatch + "_val.csv", index=False)
