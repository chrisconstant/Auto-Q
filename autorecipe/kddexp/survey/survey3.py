from random import sample
import pandas as pd

A = []
for i in range(30):
    df_filtered = pd.read_csv('user_response.csv',header=None)
    df_filtered.columns = ['id', 'uuid', 'q', 'answer']
    #print (df)

    dummy_uuid = ['a3af6958-a14e-4c88-881f-9694c40fccbc', 'd35f771e-f4be-403b-90da-eb0e61790007']
    df = df_filtered[~df_filtered['uuid'].isin(dummy_uuid)]
    df_filtered = df

    grouped = df_filtered.groupby('q')

    pairs = []
    total = 50
    # Iterate over the groups
    for name, group in grouped:
        # Select groups that have at least two answers
        if len(group) >= 2:
            # Sample pairs from the group until you have 50 pairs in total
            for _ in range(min(len(group) // 2, total - len(pairs))):
                pair = group.sample(n=2)[['answer']].T.reset_index(drop=True)
                pair.columns = ['annotator1', 'annotator2']
                pairs.append(pair)
        
        # Break the loop if you already have 50 pairs
        if len(pairs) == total:
            break
    # Concatenate the pairs into a single DataFrame
    paired_df = pd.concat(pairs)
    # Reset the index of the DataFrame
    paired_df.reset_index(drop=True, inplace=True)

    # Display the paired DataFrame
    total_accuracy = 0

    # Iterate over the rows of the paired DataFrame
    for index, row in paired_df.iterrows():
        annotator1 = row['annotator1']
        annotator2 = row['annotator2']
        
        # Check if both annotators agree
        if annotator1 == annotator2:
            total_accuracy += 1
        # Check if either annotator labels a tie
        elif "Both" in annotator1 or "Both" in annotator2:
            total_accuracy += 0.5

    # Calculate the tie-discounted accuracy
    tie_discounted_accuracy = total_accuracy / len(paired_df)
    A.append(tie_discounted_accuracy)
    print("Tie-Discounted Accuracy:", tie_discounted_accuracy)

import numpy as np
print (np.mean(A), np.std(A))

'''

dummy_uuid = ['a3af6958-a14e-4c88-881f-9694c40fccbc', 'd35f771e-f4be-403b-90da-eb0e61790007']

df = df_filtered[~df_filtered['uuid'].isin(dummy_uuid)]

print(df)

df_filtered['majority_label'] = df_filtered.groupby('q')['answer'].transform(lambda x: x.mode().iloc[0])
# Calculate the tie-discounted accuracy
total_accuracy = 0
total_data_points = len(df_filtered)

for index, row in df_filtered.iterrows():
    if row['answer'] == row['majority_label']:  # No tie
        total_accuracy += 1
    else:
        total_accuracy += 0.5

tie_discounted_accuracy = total_accuracy / total_data_points

print("Tie-Discounted Accuracy:", tie_discounted_accuracy)

shared_set = df_filtered.sample(n=50, random_state=42)

# Calculate the majority label for each data point in the shared set
shared_set['majority_label'] = shared_set.groupby('q')['answer'].transform(lambda x: x.mode().iloc[0])

# Calculate the tie-discounted accuracy over the shared set
total_accuracy = 0

for index, row in shared_set.iterrows():
    if row['answer'] == row['majority_label']:  # No tie
        total_accuracy += 1
    elif row['answer'] != row['majority_label'] and row['answer'] != "Tie" and row['majority_label'] != "Tie":  # Both annotators labeled a tie
        total_accuracy += 0.5

tie_discounted_accuracy = total_accuracy / len(shared_set)

print("Tie-Discounted Accuracy over the shared set of 50 annotation examples:", tie_discounted_accuracy)

'''