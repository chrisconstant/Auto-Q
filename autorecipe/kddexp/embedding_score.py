# %%
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.cluster import KMeans
import numpy as np

option1 = "all-MiniLM-L6-v2"  # fast
option2 = "multi-qa-mpnet-base-dot-v1"  # qa
option3 = "all-mpnet-base-v2"  # best

def get_dispersion_metric(questions, embeding_option=option1):
    model = SentenceTransformer(embeding_option)
    # Compute embeddings
    embeddings = model.encode(questions, convert_to_tensor=True)
    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)
    cosine_scores = cosine_scores * (1 - torch.eye(cosine_scores.size(0)))
    # Calculate the overall mean similarity
    overall_mean_similarity = torch.mean(cosine_scores)
    num_clusters = 10
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(embeddings)
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    # Calculate the mean distance of each sample to its cluster center
    mean_distances = np.zeros_like(cluster_assignments, dtype=np.float32)
    for i, cluster_index in enumerate(cluster_assignments):
        mean_distances[i] = np.linalg.norm(embeddings[i] - cluster_centers[cluster_index])
    # Calculate the overall mean distance
    overall_mean_distance = np.mean(mean_distances)
    returnRes = {}
    returnRes['MeanSimilarity'] = overall_mean_similarity.item()
    returnRes['MeanDistance'] = overall_mean_distance
    return returnRes

# %%
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
questions = df['question'].to_list()

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
questions = df['question'].to_list()

ans = get_dispersion_metric(questions, embeding_option=option1)
print (option1, ans)

ans = get_dispersion_metric(questions, embeding_option=option2)
print (option2, ans)

ans = get_dispersion_metric(questions, embeding_option=option3)
print (option3, ans)

#all-MiniLM-L6-v2 {'MeanSimilarity': 0.08487808704376221, 'MeanDistance': 0.8959959}



