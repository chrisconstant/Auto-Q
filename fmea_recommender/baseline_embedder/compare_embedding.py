import pandas as pd

df1 = pd.read_csv('./Mistral_sup_simcse_test.csv')
df2 = pd.read_csv('./Mistral_unsup_simcse_test.csv')
df3 = pd.read_csv('./SentenceTransformer_sup_test.csv')

data = pd.read_csv('./data/all_ids_deduplicated_train_short_desc_to_failure_locations.csv')
tdata = pd.read_csv('./data/all_ids_deduplicated_test_short_desc_to_failure_locations.csv')
L = list(data['TypeData.GenCompType'])
Lp = list(tdata['TypeData.GenCompType'])

l1 = list(df1['Top Index 1'])
l2 = list(df2['Top Index 1'])
l3 = list(df3['Top Index 1'])

for i in range(len(l1)):
    print ('------------------')
    print (Lp[i], '\n', L[l1[i]], '\n', L[l2[i]], '\n', L[l3[i]])