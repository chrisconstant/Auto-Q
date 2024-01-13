import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('./data/testResult.csv')

y_pred = df['answer'].to_list()
y_true = df['label'].to_list()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print (f'Accuracy: {accuracy}, F1: {f1}')

print("Confusion Matrix:")
print(conf_matrix)
print (np.sum(y_true))

"""
Accuracy: 0.68, F1: 0.7419354838709677
Confusion Matrix:
[[44 37]
 [27 92]]
119
75% identify
"""