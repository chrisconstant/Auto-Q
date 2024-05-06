import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
sentence_model = SentenceTransformer("all-mpnet-base-v2")

# Load data
train_df = pd.read_csv(
    "./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv"
)
test_df = pd.read_csv(
    "./data/all_ids_deduplicated_test_short_desc_to_failure_locations.csv"
)
val_df = pd.read_csv(
    "./data/all_ids_deduplicated_val_short_desc_to_failure_locations.csv"
)

# Encode train, test, and validation data
train_documents = list(train_df["TypeData.GenCompType"])
test_documents = list(test_df["TypeData.GenCompType"])
val_documents = list(val_df["TypeData.GenCompType"])

train_reps = sentence_model.encode(train_documents)
test_reps = sentence_model.encode(test_documents)
val_reps = sentence_model.encode(val_documents)

# Normalize representations
train_reps_norm = torch.nn.functional.normalize(torch.tensor(train_reps), p=2, dim=1)
test_reps_norm = torch.nn.functional.normalize(torch.tensor(test_reps), p=2, dim=1)
val_reps_norm = torch.nn.functional.normalize(torch.tensor(val_reps), p=2, dim=1)

# Calculate cosine similarity
train_cos_sim = torch.mm(train_reps_norm, train_reps_norm.transpose(0, 1))
test_cos_sim = torch.mm(test_reps_norm, train_reps_norm.transpose(0, 1))
val_cos_sim = torch.mm(val_reps_norm, train_reps_norm.transpose(0, 1))


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


process_cos_sim(test_cos_sim, "SentenceTransformer_sup_test")
process_cos_sim(val_cos_sim, "SentenceTransformer_sup_val")
