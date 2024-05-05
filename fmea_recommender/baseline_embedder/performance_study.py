import pandas as pd
import torch
import random

from genai import Client
from genai.schema import TextGenerationParameters
from sentence_transformers import SentenceTransformer, util as st_util


### Generate Prompts
def shuffle_examples(eg_inputs, eg_outputs, permutation):
    """Shuffle the input and output components of a shot according to permutation"""
    shuffled_eg_inputs = []
    shuffled_eg_outputs = []
    for i in permutation:
        shuffled_eg_inputs.append(eg_inputs[i])
        shuffled_eg_outputs.append(eg_outputs[i])
    return shuffled_eg_inputs, shuffled_eg_outputs

def setup_few_shot_retrieval(train_data_path, query_column):
    """Setup dynamic shot retrieval.
    query_column - column to encode.
     - generate embeddings for training methods"""
    rag_model = SentenceTransformer('all-mpnet-base-v2')
    train_data = pd.read_csv(train_data_path)
    train_examples = list(train_data[query_column])
    # Compute embeddings
    train_embeddings = rag_model.encode(train_examples, convert_to_tensor=True)
    dynamic_shots_cfg = {
        "train_data" : train_data,
        "train_embeddings" : train_embeddings,
        "rag_model" : rag_model
    }
    return dynamic_shots_cfg


def get_bam_config(credentials,
               decoding_method="greedy",
               max_new_tokens=200,
               repetition_penalty=1.2):
    """Configure BAM Model"""
    client = Client(credentials=credentials)

    params = TextGenerationParameters(decoding_method=decoding_method,
                            max_new_tokens=max_new_tokens,
                            repetition_penalty=repetition_penalty)
    bam_config = {'client' : client,
                  'generate_params' : params}
    return bam_config

def get_related_examples(inputs, dynamic_shots_cfg, max_num_egs_in_prompt):
    """Given the inputs retrieve the most related examples from the configured colletion.
    If (max_num_egs_in_prompt == -1) then we include a randomly selected example,
    otherwise we include the most related examples.
    Output: A list with of the indices (positions) of the most related examples (or a random example if requested)
    """
    inputs_embeddings = dynamic_shots_cfg["rag_model"].encode(inputs, convert_to_tensor=True)
    comparison_results = st_util.dot_score(
        inputs_embeddings, dynamic_shots_cfg["train_embeddings"]
    )
    if max_num_egs_in_prompt == -1:  # Include a random example.
        best_k_matches = []
        min = 0
        max = len(dynamic_shots_cfg["train_data"]) - 1
        for _ in inputs:
            best_k_matches.append([random.randint(min, max)])
    else:
        best_k_matches = torch.topk(
            comparison_results, max_num_egs_in_prompt, dim=1
        ).indices.tolist()
    return best_k_matches
