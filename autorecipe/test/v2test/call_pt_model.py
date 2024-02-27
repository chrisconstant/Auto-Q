from rouge_score import rouge_scorer
import ray
import numpy as np
import nltk as nlp
import re
from nltk.probability import FreqDist
import math
from genai.credentials import Credentials
from genai.schema import (
    DecodingMethod,
    TextGenerationParameters,
)
from genai.client import Client
from datasketch import MinHashLSH, MinHash

def call_Question(sentence):

    api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
    api_url = "https://bam-api.res.ibm.com"
    creds = Credentials(api_key, api_endpoint=api_url)
    print("\n------------- Example (Model Talk)-------------\n")
    bob_params = TextGenerationParameters(decoding_method=DecodingMethod.GREEDY, 
                                          max_new_tokens=100, 
                                          temperature=1)
    client = Client(credentials=creds)
    print(f"classify { 'DFM', 'ELK' } Input: {sentence} Output:")
    q_response =  next(client.text.generation.create(model_id="flan-t5-xl-pt-W2z9ge8Q-2024-02-27-06-08-45",
                                                inputs=[sentence],
                                                parameters=bob_params,))
    q_gen = q_response.results[0]
    print (q_gen)

call_Question('MUD PUMP - Replace HMI, COLOR, 3IN, HORNER, HEP')



