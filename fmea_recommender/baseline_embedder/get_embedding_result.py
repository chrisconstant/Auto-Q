import pandas as pd
import torch
from llm2vec import LLM2Vec

l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv('./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv')
documents = list(df['TypeData.GenCompType'])
d_reps = l2v.encode(documents)

df = pd.read_csv('./data/all_ids_deduplicated_test_short_desc_to_failure_locations.csv')
querys = list(df['TypeData.GenCompType'])
instruction = (
    "Given an industrial equipment, retrieve similar equipment:"
)
queries = [
    instruction + " " + item for item in querys
]

q_reps = l2v.encode(queries)

# passing the results
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

top_values, top_indices = torch.topk(cos_sim, k=3, dim=1)

print (top_values)
print (top_indices)
