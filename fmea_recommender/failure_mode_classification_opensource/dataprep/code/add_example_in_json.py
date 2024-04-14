import pandas as pd
import json

# Load the JSON file
with open("../metadata.json", "r") as f:
    data = json.load(f)

df = pd.read_csv("../train.csv")

# Iterate over each dictionary in the list
for item in data:
    # Add a new key with a list of strings as the value
    item["examples"] = list(df[df["label"] == item["label"]]["workordertext"])
    print (item)

# Save the modified data back to a JSON file
with open("final_metadata.json", "w") as f:
    json.dump(data, f, indent=4)
