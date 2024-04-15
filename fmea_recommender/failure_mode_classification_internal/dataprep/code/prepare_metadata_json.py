import pandas as pd

df = pd.read_csv("../data/client_1_train_trial.csv")

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

failurelbl = pd.read_csv('../data/failure_code.csv')

df = df[selected_columns]
df.columns = rename_columns

merged_df = pd.merge(failurelbl, df, on='label', how='left')
df = merged_df

grouped = (
    df.groupby(["label", "description", "longdescription"])
    .apply(
        lambda x: x.drop(columns=["label", "description", "longdescription"]).to_dict(
            orient="records"
        )
    )
    .reset_index(name="examples")
)


# Create a list of dictionaries containing 'label', 'description', and 'examples'
result = []
for index, row in grouped.iterrows():
    group_dict = {
        "label": row["label"],
        "description": row["description"],
        "longdescription": row["longdescription"],
        "examples": row["examples"],
    }
    result.append(group_dict)

import json

with open("../../metadata.json", "w") as f:
    json.dump(result, f, indent=4)
