autoais_model_dir = "/dccstor/dcpfactory"
autoais_tokenizer_dir = "/dccstor/dcpfactory"

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)

import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
import re
import torch
import numpy as np

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"

def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    if inference == 1:
        print (result, input_text)
    return inference

def compute_claims(data):
    global autoais_model, autoais_tokenizer
    autoais_model = None
    autoais_tokenizer = None
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto", cache_dir=autoais_model_dir)
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False, cache_dir=autoais_model_dir)

    logger.info("Computing claims...")
    scores = []
    for item in tqdm(data):
        normalized_output = remove_citations(item['output'])
        entail = 0
        claims = item["claims"]
        for claim in claims:
            entail += _run_nli_autoais(normalized_output, claim)
        scores.append(entail / len(claims))
    return 100 * np.mean(scores)

#print (compute_claims(data))
import pandas as pd
import ast
dff = pd.read_csv('combined_file_wind_turbine.csv')

window_size = 5  # Define the size of the sliding window

# Loop through the DataFrame rows
for i in range(dff.shape[0]):
    x = dff.docs[i]
    xx = ast.literal_eval(x)

    # Slide the window by 1
    for k in range(len(xx) - window_size + 1):
        # Concatenate 5 consecutive items into one paragraph
        paragraph = ' '.join(xx[k:k+window_size])
        # Create the test input
        one_test = [{'output': paragraph, 'claims': [dff.passageclaim[i]]}]
        # Compute claims for the test input
        compute_claims(one_test)
