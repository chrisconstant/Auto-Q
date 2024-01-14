from __future__ import print_function
print(__doc__)

import numpy as np
import time

import spacy
from grakel import GraphKernel, Graph

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def generate_dependency_graph(sentence):
    # Parse sentence and construct dependency tree using spaCy
    doc = nlp(sentence)
    
    # Transform dependency tree to grakel Graph representation
    edges = []
    node_labels = dict()

    for token in doc:
        if token.dep_ != 'punct':  # Exclude punctuation dependencies
            head_node = (token.head.text, token.head.i)
            dep_node = (token.text, token.i)
            
            edges.append((head_node, dep_node, {'relation': token.dep_}))
            node_labels[head_node] = token.head.text
            node_labels[dep_node] = token.text
    
    return Graph(edges, node_labels=node_labels)

def find_most_similar_sentence(target_sentence, sentences):
    # Generate dependency graph for the target sentence
    target_graph = generate_dependency_graph(target_sentence)
    
    # Generate dependency graphs for all sentences
    all_graphs = [generate_dependency_graph(sentence) for sentence in sentences]

    # Apply Weisfeiler-Lehman graph kernel
    gk = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5})
    kernel_matrix = gk.fit_transform(all_graphs)

    # Calculate similarity scores
    similarity_scores = kernel_matrix[:, sentences.index(target_sentence)].reshape(-1)

    # Exclude similarity to itself
    similarity_scores[sentences.index(target_sentence)] = 0.0

    # Find the index of the most similar sentence
    most_similar_index = np.argmax(similarity_scores)
    
    return sentences[most_similar_index]

# Example usage
sentences = ["Which magazine was started first Arthur's Magazine or First for Women?",
             "The Oberoi family is part of a hotel company that has a head office",
             "In what city? Musician and satirist Allie Goertz wrote a song about the The Simpsons character Milhouse, who Matt Groening named after who?"]

for target_sentence in sentences:
    most_similar_sentence = find_most_similar_sentence(target_sentence, sentences)
    print(f"Target Sentence: {target_sentence}")
    print(f"Most Similar Sentence: {most_similar_sentence}\n")
