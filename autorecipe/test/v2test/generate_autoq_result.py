import os
import pandas as pd

def get_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename.startswith('genai_context_docs'):
            csv_files.append(os.path.join(directory, filename))
    return csv_files

directory_path = "./experiment1_contextonly_withboundry/"
csv_files_list = get_csv_files(directory_path)
print(len(csv_files_list))

gold_df = pd.read_csv("./autoQ_val_data_input_for_experiments.csv")
trial = list(gold_df["component_short_description"])

for item in trial:
    item = item.replace('<', '-')
    item = item.replace('/', ' ').replace(',',' ')
    item = item.replace(' ', '')
    notfound = True
    filename = ''
    for name in csv_files_list:
        if item in name:
            notfound = False
            filename = name
            break

    if notfound:
        pass
    else:
        print (item, filename)
