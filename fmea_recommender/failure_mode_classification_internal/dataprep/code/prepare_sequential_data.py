import pandas as pd

df = pd.read_csv("../data/client_1_test_trial.csv")

selected_columns = [
    "description",
    "persongroup",
    "crewworkgroup",
    "assetnum",
    "asset description",
    "location",
    "location description",
    "worktype",
    "siteid",
    "problemcode",
    "taskid",
    "task description",
]

rename_columns = [
    "workorderdescription",
    "persongroup",
    "crewworkgroup",
    "assetnumer",
    "assetdescription",
    "location",
    "locationdescription",
    "workordertype",
    "siteid",
    "label",
    "taskid",
    "workordertaskdescription",
]

df = df[selected_columns]
df.columns = rename_columns

output_json = []
for _, row in df.iterrows():
    entry = {}
    entry['input'] = row.drop(['label']).to_dict()
    entry['output'] = row['label']
    output_json.append(entry)

import json
# Write JSON to file
with open('../../sequencedata.json', 'w') as f:
    json.dump(output_json, f, indent=4)