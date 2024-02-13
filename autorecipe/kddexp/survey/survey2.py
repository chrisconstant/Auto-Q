from random import sample

import pandas as pd
df_filtered = pd.read_csv('user_survey_1.csv',header=None)
df_filtered.columns = ['id', 'uuid', 'q', 'answer']
#print (df)

dummy_uuid = ['a3af6958-a14e-4c88-881f-9694c40fccbc', 'd35f771e-f4be-403b-90da-eb0e61790007']

df = df_filtered[~df_filtered['uuid'].isin(dummy_uuid)]

print(df)

answer_counts = df.groupby(['q', 'answer']).size().unstack(fill_value=0)
print (answer_counts)

# Determine the label with the majority vote for each question
minority_labels = answer_counts.idxmin(axis=1)
print (minority_labels)

second_highest_labels = answer_counts.apply(lambda x: x.sort_values(ascending=False).index[1], axis=1)
print(second_highest_labels)

minority_labels_sorted = minority_labels.sort_index()
second_highest_labels_sorted = second_highest_labels.sort_index()

print (minority_labels)

exit(0)

df1 = pd.read_csv('../../../survey/questions.csv')
Qid = ['Q' + str(i+1) for i in range(20)]
df1['q'] = Qid
df1 = df1[['Real Persona','q']]
#print (df1)

def match(x, y):
    if x == y:
        return 1
    else:
        return 0
    '''
    elif x == "Subject Matter Expert" and "Both" in y:
        return 0.5
    elif x == "Data Scientist" and "Both" in y:
        return 0.5
    else:
        return 0
    '''
merged_table = df.merge(df1, on='q')
merged_table['binary'] = merged_table.apply(lambda row: match(row['answer'], row['Real Persona']), axis=1)
print (merged_table['binary'].sum())
print (merged_table)

selected_q = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7']
merged_table['subset'] = merged_table['q'].apply(lambda x: 0 if x in selected_q else 1)
merged_table = merged_table[merged_table['subset']==1]
print (merged_table['binary'].sum())

# question wise correctness
print (merged_table.groupby('q').sum('binary'))

# userwise wise correctness
print (merged_table.groupby('uuid').sum('binary'))

merged_table['annotator_1'] = merged_table['binary']
merged_table['annotator_2'] = merged_table['binary'].sample(frac=1).reset_index(drop=True)  # Simulating a different annotator

# Calculate tie-discounted accuracy with partial agreement
total_weighted_agreement = 0
total_pairs = 0

for response_1, response_2 in zip(merged_table['annotator_1'], merged_table['annotator_2']):
    if response_1 == response_2:
        total_weighted_agreement += 1
    else:
        weight = min(response_1, response_2)
        total_weighted_agreement += weight
    total_pairs += 1

tie_discounted_accuracy = total_weighted_agreement / total_pairs * 100

print(f'Tie-Discounted Accuracy (with partial agreement): {tie_discounted_accuracy:.2f}%')
