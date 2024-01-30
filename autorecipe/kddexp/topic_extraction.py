# %%
import pandas as pd
df = pd.read_csv('../test/genai_questions_809156896092057937.csv')
#df = pd.read_csv('../test/genai_questions_311016922297059474.csv')
df = pd.read_csv("../../results/windturbinegearbox/genai_questions_bank_windturbinegearbox_llama.csv")
questions = df['questions'].to_list()
print (len(questions))

# %%
from datasets import load_dataset

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# We select a subsample of 5000 abstracts from ArXiv
docs = questions

# We define a number of topics that we know are in the documents
zeroshot_topic_list = ['failures', 'components']

# We fit our model using the zero-shot topics
# and we define a minimum similarity. For each document,
# if the similarity does not exceed that value, it will be used
# for clustering instead.
topic_model = BERTopic(
    embedding_model="thenlper/gte-large", 
    min_topic_size=15,
    zeroshot_topic_list=zeroshot_topic_list,
    zeroshot_min_similarity=.85,
    representation_model=KeyBERTInspired()
)
topics, _ = topic_model.fit_transform(docs)


# %%
ans = topic_model.get_topic_info()

# %%
topicans = ans['Representation'].to_list()

# %%
for item in topicans:
    print (item)

# %%
ans['Representative_Docs'][0]

# %%
ans.to_csv('ResTopic.csv',index=False)


