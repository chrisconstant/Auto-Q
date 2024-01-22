# %%
import pandas as pd
import numpy as np
import copy
from nltk.translate.bleu_score import sentence_bleu

# Suggestion
"""
in percentage
< 10 	Almost useless
10 - 19 	Hard to get the gist
20 - 29 	The gist is clear, but has significant grammatical errors
30 - 40 	Understandable to good translations
40 - 50 	High quality translations
50 - 60 	Very high quality, adequate, and fluent translations
> 60 	Quality often better than human
"""
# https://cloud.google.com/translate/automl/docs/evaluate

def get_bleu_score(remaining_sentences, sentence):
    ans = sentence_bleu(remaining_sentences, sentence)
    return ans

def calculate_selfBleu(sentences):
    bleu_scores = []
    for i in sentences:
        sentences_copy = copy.deepcopy(sentences)
        sentences_copy.remove(i)
        bleu_scores.append(get_bleu_score(sentences_copy, i))
    return np.mean(bleu_scores)

df = pd.read_csv('../test/genai_questions_809156896092057937.csv')
questions = df['questions'].to_list()
questions = list(set(questions))
sentences = [item.split() for item in questions]
ans = calculate_selfBleu(sentences)
print (ans)


"""
df = pd.read_csv('./existing_qa_datasets/squad_v2_q.csv')
questions = df['question'].to_list()
ans = calculate_selfBleu(questions)
print (ans)

df = pd.read_csv('./existing_qa_datasets/hotpot_q.csv')
questions = df['question'].to_list()
ans = calculate_selfBleu(questions)
print (ans)
"""

# %%
# pip install accelerate auto-gptq optimum
# too slow
from transformers import LlamaForCausalLM, LlamaTokenizerFast
import torch
from transformers import AutoConfig, AutoModelForCausalLM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained("TheBloke/Llama-2-13B-GPTQ")
config.quantization_config["disable_exllama"] = True


model = LlamaForCausalLM.from_pretrained(
    "TheBloke/Llama-2-13B-GPTQ", torch_dtype=torch.bfloat16, device_map=device, config=config
).to(device)
tokenizer = LlamaTokenizerFast.from_pretrained("TheBloke/Llama-2-13B-GPTQ")

inputs = tokenizer("ABC is a startup based in New York City and Paris", return_tensors = "pt")
loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
ppl = torch.exp(loss)
print (ppl)

sentence1 = "ABC is a startup based in New York City and Paris"
sentence2 = "Generative Pretrained Transformer is an opensource artificial intelligence created by OpenAI in February 2019"

inputs = tokenizer(sentence1, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}
loss = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"]).loss
ppl = torch.exp(loss)
print("Sentence 1 PPL:", ppl.item())

inputs_wiki_text = tokenizer(sentence2, return_tensors="pt")
inputs_wiki_text = {key: value.to(device) for key, value in inputs_wiki_text.items()}
loss = model(input_ids=inputs_wiki_text["input_ids"], labels=inputs_wiki_text["input_ids"]).loss
ppl = torch.exp(loss)
print("Sentence 2 PPL:", ppl.item())



