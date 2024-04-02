#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import ray

from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions,
)
from dotenv import load_dotenv


# In[4]:


load_dotenv()
client = Client(credentials=Credentials.from_env())


# In[5]:


df_train = pd.read_csv('../fmea_result/autoQ_train_data_input_for_experiments.csv')
df_val = pd.read_csv('../fmea_result/autoQ_val_data_input_for_experiments.csv')
df_test = pd.read_csv('../fmea_result/autoQ_test_data_input_for_experiments.csv')


# In[7]:


df_train.columns


# In[8]:


type_col = 'TypeData.GenCompType'
long_desc_col = 'long_description'
failure_loc_col = 'failure_locations'


# In[11]:


template = f"""
I have an industrial asset, its long description and the places where the asset can fails. Your task is to generate a redable summary of failure locations only using provided information in failure location. 
Asset: <asset>
Long Description: <long_desc>
Failure Locations: <failure_loc>
Here is a readable Summary of Failure Locations:
"""


# In[61]:


def get_prompts(df):
    prompts = []
    for idx, row in df.iterrows():
        asset = row[type_col]
        long_desc = row[long_desc_col]
        failure_loc = str(eval(row[failure_loc_col])).replace('{', '').replace('}', '').replace("'", "")
        prompt = template.replace('<asset>', asset) \
                         .replace('<long_desc>', long_desc) \
                         .replace('<failure_loc>', failure_loc)
        prompts.append(prompt)
    return prompts


# In[62]:


train_prompts = get_prompts(df_train)
val_prompts = get_prompts(df_val)
test_prompts = get_prompts(df_test)


# In[68]:

ray.init(num_cpus=8)

@ray.remote
def get_summaries(prompt):
    results = client.text.generation.create(
        model_id="mistralai/mixtral-8x7b-instruct-v0-1",
        inputs=[prompt],
        parameters=TextGenerationParameters(
            return_options=TextGenerationReturnOptions(
                input_text=True,
            ),
        ),
    )
    return results[0].results[0]


# In[ ]:

remote_call = []
for pt in train_prompts:
    remote_call.append(get_summaries.remote(pt))
train_summaries = ray.get(remote_call)
print('train generated')

remote_call = []
for pt in val_prompts:
    remote_call.append(get_summaries.remote(pt))
val_summaries = ray.get(remote_call)
print('val generated')

remote_call = []
for pt in test_prompts:
    remote_call.append(get_summaries.remote(pt))
test_summaries = ray.get(remote_call)
print('test generated')


# In[ ]:




