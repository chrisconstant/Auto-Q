import mlflow
import os
import json
from extract_questions_from_string import extract_fun
from itertools import chain

experiments = mlflow.search_experiments()

eSet = ['657317679692709876']

orders = []

for exp in experiments:
    if exp.experiment_id in eSet:
        #print (exp)
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time asc"])
        for _, run in runs.iterrows():
            run_id = run.run_id
            timestamp = run.start_time
            status = run.status
            #print (run.artifact_uri)

            fname = (run.artifact_uri + "/" + "Answer.json").split('file:///')
            try:
                with open(fname[1], 'r') as json_file:
                    json_content_text = json.load(json_file)
                    if 'questions' in json_content_text['Answer']:
                        qq = extract_fun(json_content_text['Answer'])
                        if len(qq) > 0:
                            orders.append(qq)
                        else:
                            print (json_content_text['Answer'])
            except:
                pass


flat_list = list(chain(*orders))
print (flat_list)
print (len(flat_list))

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

total_sim = 0
for i in range(len(flat_list)):
    for j in range(len(flat_list)):
        if i < j:
            score = scorer.score(flat_list[i],flat_list[j])
            if score['rougeL'].fmeasure > 0.98:
                print ('source ..... ---->', score['rougeL'].fmeasure)
                print (flat_list[i], '\n', flat_list[j])
                total_sim = total_sim + 1

print (total_sim)
import pandas as pd
df = pd.DataFrame({'questions': flat_list})
df.to_csv('genai_questions_1.csv', index=False)
