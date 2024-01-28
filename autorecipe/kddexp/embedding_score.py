# %%
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.cluster import KMeans
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

option1 = "all-MiniLM-L6-v2"  # fast
option2 = "multi-qa-mpnet-base-dot-v1"  # qa
option3 = "all-mpnet-base-v2"  # best

from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.cluster import KMeans
import numpy as np

option1 = "all-MiniLM-L6-v2"  # fast
option2 = "multi-qa-mpnet-base-dot-v1"  # qa
option3 = "all-mpnet-base-v2"  # best

def get_dispersion_metric(questions, embeding_option=option1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(embeding_option).to(device)
    # Compute embeddings
    embeddings = model.encode(questions, convert_to_tensor=True).to(device)
    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)
    # Set diagonal elements to zero to avoid self-similarity
    cosine_scores = cosine_scores * (1 - torch.eye(cosine_scores.size(0), device=device))
    # Calculate the overall mean similarity
    overall_mean_similarity = torch.mean(cosine_scores)
    num_clusters = 10
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(embeddings.cpu().detach().numpy())
    # Get cluster centers
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    # Calculate the mean distance of each sample to its cluster center
    mean_distances = torch.norm(embeddings - cluster_centers[cluster_assignments], dim=1)
    # Calculate the overall mean distance
    overall_mean_distance = torch.mean(mean_distances).item()
    return {'MeanSimilarity': overall_mean_similarity.item(), 'MeanDistance': overall_mean_distance}


# %%
"""
import pandas as pd
df = pd.read_csv("../test/genai_questions_809156896092057937.csv")
df = pd.read_csv("../test/genai_questions_311016922297059474.csv")
questions = df["questions"].to_list()
unique_questions = list(set(questions))
questions = unique_questions

ans = get_dispersion_metric(questions, embeding_option=option1)
print (option1, ans)

ans = get_dispersion_metric(questions, embeding_option=option2)
print (option2, ans)

ans = get_dispersion_metric(questions, embeding_option=option3)
print (option3, ans)


# %%
df = pd.read_csv('./existing_qa_datasets/squad_v2_q.csv')
for i in range(5):
    questions = df['question'].sample(n=50000, random_state=42+i).to_list()
    ans = get_dispersion_metric(questions, embeding_option=option1)
    print (option1, ans)
    ans = get_dispersion_metric(questions, embeding_option=option2)
    print (option2, ans)
    ans = get_dispersion_metric(questions, embeding_option=option3)
    print (option3, ans)

# all-MiniLM-L6-v2 {'MeanSimilarity': 0.07852756977081299, 'MeanDistance': 0.91678}
# multi-qa-mpnet-base-dot-v1 {'MeanSimilarity': 0.33204522728919983, 'MeanDistance': 4.3992586}
# all-mpnet-base-v2 {'MeanSimilarity': 0.11763931810855865, 'MeanDistance': 0.8907894}
# %%

df = pd.read_csv('./existing_qa_datasets/hotpot_q.csv')
for i in range(5):
    questions = df['question'].sample(n=50000, random_state=42+i).to_list()
    ans = get_dispersion_metric(questions, embeding_option=option1)
    print (option1, ans)
    ans = get_dispersion_metric(questions, embeding_option=option2)
    print (option2, ans)
    ans = get_dispersion_metric(questions, embeding_option=option3)
    print (option3, ans)

#all-MiniLM-L6-v2 {'MeanSimilarity': 0.08487808704376221, 'MeanDistance': 0.8959959}
"""

df = pd.read_csv('../test/genai_questions_bank_windturbinegearbox_llama.csv')
questions = df["questions"].to_list()
unique_questions = list(set(questions))
questions = unique_questions

ans = get_dispersion_metric(questions, embeding_option=option1)
print (option1, ans)

ans = get_dispersion_metric(questions, embeding_option=option2)
print (option2, ans)

ans = get_dispersion_metric(questions, embeding_option=option3)
print (option3, ans)
