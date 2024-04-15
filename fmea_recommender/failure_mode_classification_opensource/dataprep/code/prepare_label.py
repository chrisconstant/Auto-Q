import pandas as pd

df = pd.read_csv('../FMC_code_mapping.csv')
print (df)

mapping = {}
for index, row in df.iterrows():
    mapping[row['description']] = row['label']

train = pd.read_csv('../all.csv')
print (train)

train.columns = ['workordertext','description']

def check_string(x):
    if x in mapping.keys():
        return mapping[x]
    else:
        'NA'

train['label'] = train['description'].apply(lambda x: check_string(x))
print (train)

train.to_csv('../all.csv',index=False)