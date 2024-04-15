import pandas as pd
import json

# Example DataFrame
df = pd.read_csv('../test.csv')
df.columns = ['input','desc','output']
df = df[['input','output']]

# Convert DataFrame to a list of dictionaries
json_data = df.to_dict(orient='records')

# Save the data to a JSON file
with open('sequence_data_test.json', 'w') as f:
    json.dump(json_data, f, indent=4)