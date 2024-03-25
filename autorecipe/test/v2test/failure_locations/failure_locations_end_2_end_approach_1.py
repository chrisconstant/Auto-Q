import ray
from sentence_transformers import SentenceTransformer
from generate_failure_locations_from_context_thinking_components import get_components
from generate_failure_locations_from_context_thinking_failure_mode import (
    get_failure_modes,
)
from generate_failure_locations_from_context_thinking_failure_location import (
    get_failure_locations,
)
from validation import extract_things_from_string, calculate_precision, calculate_recall

import pandas as pd
import os

import nltk
from nltk.stem import WordNetLemmatizer


def get_lemmatized_components(components):
    lemmatizer = WordNetLemmatizer()
    lemmatized_components = [
        lemmatizer.lemmatize(component.lower()) for component in components
    ]
    unique_lemmatized_components = list(set(lemmatized_components))
    sorted_components = sorted(unique_lemmatized_components)

    similar_components = []
    for i in range(len(sorted_components) - 1):
        if sorted_components[i] == sorted_components[i + 1][:-1] and sorted_components[i + 1][-1] == 's':
            similar_components.append(sorted_components[i + 1])

    sorted_components = [component for component in sorted_components if component not in similar_components]
    return sorted_components


"""
Hydraulic Cylinder - Seals  --->  Space - Space ---> component followed by sub-component
Dust Cover, if present      ---> we should remove "if present"
Note: Some devices will     ---> we should remove this part
Rotor Cage - lamination support, i.e. spider  ---> this one also need
Stator Windings, includes Blocking, Bracing, Tying, Surge rings, Wedges ---> includes (break)
Enclosure (Cubicle & Breaker)  ---> ??
Bearing Seals (all types)' ---> ??

"""

"""
'Documentation'
'Records', 
'Design Specifications'
'Test Reports'
'Maintenance Records'
'Failures' (any thing that end with)
'Other Components'
(if present)
Miscellaneous Components
Other Components
Hydraulic Cylinder - Seals' (space - space --> parent to child components)
two words with (s) or (es) lemmanization
Model Number
Base
SCADA Server/Client
Miscellaneous Components
Other Components
National and International Standards
FMEA
Efficiency
Installation

"""

cross_check_ans = [
    "I apologize",
    "I'm sorry",
    "I am sorry",
    "I don't understand",
]


def get_all_names(data):
    all_names = []
    all_subcomponents = []

    for item in data:
        all_names.append(item["name"])
        all_subcomponents.extend(item["subcomponents"])

    all_names.extend(all_subcomponents)
    unique_names = list(set(all_names))
    return unique_names

ray.init(num_cpus=8)

@ray.remote
def get_evaluation(model_id, asset_class, asset_class_result_file, gtruth):
    """ """
    val_model = SentenceTransformer("all-mpnet-base-v2")
    df = pd.read_csv(asset_class_result_file)
    dclm = df.columns
    dclm = dclm[:-1]
    A = df.to_numpy()
    final_results = []
    ret_result = [asset_class]

    for i in range(len(dclm)):
        sample_assetdesc = A[0, i]

        quality_check = True
        for word in cross_check_ans:
            if word in sample_assetdesc:
                quality_check = False

        if not quality_check:
            ret_result.extend([dclm[i], None, None])
        else:
            print(asset_class, sample_assetdesc)
            components_str, component_list = get_components(
                asset_class, sample_assetdesc, model_id=model_id
            )
            failuremode_ans = get_failure_modes(components_str, model_id=model_id)
            failure_locations = get_failure_locations(
                failuremode_ans, model_id=model_id
            )
            failure_locations.extend(component_list)
            results = get_all_names(failure_locations)
            results = get_lemmatized_components(results)
            final_results.extend(results)
            pres = calculate_precision(results, gtruth, val_model)
            rres = calculate_recall(results, gtruth, val_model)
            ret_result.extend([dclm[i], pres, rres])
            print(dclm[i], pres, rres)

    unique_names = get_lemmatized_components(final_results)
    pres = calculate_precision(unique_names, gtruth, val_model)
    rres = calculate_recall(unique_names, gtruth, val_model)
    ret_result.extend([unique_names, pres, rres])

    return ret_result


def get_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith("genai_context_docs"):
            csv_files.append(os.path.join(directory, filename))
    return csv_files

directory_paths = [
                   ("./experiment1_contextonly_withboundry/mixtral/",5),
                   ("./experiment1_contextonly_withboundry/granite/",3),
                   ("./experiment1_contextonly_withboundry/lamma/",1)]

LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v01-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
]

for directory_path, model_id in directory_paths:
    # will repeat the 

    csv_files_list = get_csv_files(directory_path)

    gold_df = pd.read_csv("./autoQ_val_data_input_for_experiments.csv")
    asset_classes = list(gold_df["component_short_description"])
    goldenset = gold_df["failure_locations"]

    refs = []
    for item_index, item in enumerate(asset_classes):
        item = item.replace("<", "-")
        item = item.replace("/", " ").replace(",", " ")
        item = item.replace(" ", "")
        notfound = True
        filename = ""
        for name in csv_files_list:
            if item in name:
                notfound = False
                filename = name
                break

        if notfound:
            pass
        else:
            print(item, filename)
            gtruth = extract_things_from_string(goldenset[item_index])
            print(gtruth)
            try:
                refs.append(get_evaluation.remote(model_id, item, filename, gtruth))
            except:
                pass
            # refs.append(get_evaluation.remote(model_id, item, filename))

    parallel_returns = ray.get(refs)
    # print (parallel_returns)

    res = pd.DataFrame(parallel_returns)
    model_initial = LLMsets[model_id].split("/")[1].split("-")[0]
    directory_path = directory_path.replace("/", "").replace(".", "")
    res.to_csv(
        f"autoQ_context_guided_pipeline_generated_result_{model_initial}_{directory_path}.csv",
        index=False,
    )
