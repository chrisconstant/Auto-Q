import csv
import json

def csv_to_json(csv_file, json_file):
    with open(csv_file, 'r') as file:
        csv_data = csv.DictReader(file)
        
        data_list = list(csv_data)
        
    # Write the JSON data to a file
    with open(json_file, 'w') as file:
        # Convert the data list to JSON and write it to the file
        json.dump(data_list, file, indent=4)

# Replace 'input.csv' and 'output.json' with your file paths
csv_to_json('../FMC_code_mapping.csv', '../metadata.json')
