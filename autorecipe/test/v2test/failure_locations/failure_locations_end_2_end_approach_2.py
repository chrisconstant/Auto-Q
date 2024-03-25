from autorecipe.genai.GenAIChat import GenAIChatClient
import uuid
import mlflow
from dotenv import load_dotenv
import uuid
import mlflow
import pandas as pd
from validation import extract_things_from_string, calculate_precision, calculate_recall
import pandas as pd
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationReturnOptions,
)
import ray

def get_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith("genai_context_docs"):
            csv_files.append(os.path.join(directory, filename))
    return csv_files


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


LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v01-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
]

DEFAULT_CONFIG = {
    "model": LLMsets[3],
    "params": {
        "decoding_method": DecodingMethod.GREEDY,
        "min_new_tokens": 200,
        "max_new_tokens": 2000,  # 1500,
        "stop_sequences": ["(TOKENSTOP)"],
        "return_options": TextGenerationReturnOptions(
            input_text=False, input_tokens=True
        ),
        "moderations": ModerationParameters(
            hap=ModerationHAP(input=True, output=False, threshold=0.01)
        ),
    },
    "creds": {
        "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
        "api_endpoint": "https://bam-api.res.ibm.com",
    },
    "stream": True,
}

load_dotenv()
api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_url = "https://bam-api.res.ibm.com"

SystemPromptTemplate = """
You are helpful assistant. User will provide you an industrial asset with its description. Your taks is 
to generate the failure locations for each of the component or subcomponent. The failure locations is a 
physical or logical component/part of that asset's component or subcomponent.

Asset Class: {asset_class}

Asset Description: {asset_description}

"""

promptSeq = [
    """Generate all the failure modes of {asset_class}. Provide comprehensive answers along with highlights 
of all the failure locations and the name of failed component at that location. Your answer should covers 
as many as failure mode, locations and failed component. Include electrical, 
mechanical, thermal and others factors that affect failure mode if any. Increase coverage of answers as 
well as include name of specific component/subcomponent or some add-on components or boundry components. 
Provide more comprehensive and accurate list of the specific components that can fail in a {asset_class}.
    """,
    """Generate better solution of previous response in terms of completeness.""",
    """Generate all the failure locations including specific components failure reported in previous two response as a list. Do not generate explanation 
    or additional information. """,
    """extract all named entity representing a component or a subcomponent or a part in the previous response. 
    Only return the named entiry. Do not generate explanation or additional information or Note.""",
]

ray.init(num_cpus=8)

cross_check_ans = [
    "I apologize",
    "I'm sorry",
    "I am sorry",
    "I don't understand",
]

@ray.remote
def get_evaluation(model_id, asset_class, asset_class_result_file, gtruth):
    val_model = SentenceTransformer("all-mpnet-base-v2")
    df = pd.read_csv(asset_class_result_file)
    dclm = df.columns
    dclm = dclm[:-1]
    A = df.to_numpy()
    final_results = [asset_class]
    all_answers = []
 
    experiment_name = "MyExperiment_" + asset_class + "_" + str(uuid.uuid4())
    experiment_id = mlflow.create_experiment(experiment_name)
    stateful = True
 
    for i in range(len(dclm)):

        sample_assetdesc = A[0, i]

        quality_check = True
        for word in cross_check_ans:
            if word in sample_assetdesc:
                quality_check = False

        if not quality_check:
            final_results.extend([dclm[i], None, None])
        else:
            SystemPrompt = SystemPromptTemplate.format(
                asset_class=asset_class, asset_description=sample_assetdesc
            )
            mdl = GenAIChatClient(
                name="ADesc",
                description="Asset Description",
                skill="Generate Asset Description",
                model=LLMsets[model_id],
                params=DEFAULT_CONFIG["params"],
                system_message=SystemPrompt,
                credentials=DEFAULT_CONFIG["creds"],
                stateful=stateful,
                stream=True,
            )

            for pt in promptSeq:
                fpt = pt.format(asset_class=asset_class)
                ans = mdl.create(
                    context=None,
                    messages=[{"content": fpt, "role": "user"}],
                    experiment_id=experiment_id,
                )

            ans = ans.strip()
            print (ans)
            if 'Note:' in ans:
                ans = ans.split('Note:')[0]
            if 'These are' in ans:
                ans = ans.split('These are')[0]
            if 'Note that' in ans:
                ans = ans.split('Note that')[0]
            if 'I hope' in ans:
                ans = ans.split('I hope')[0]
            if 'INST:' in ans:
                ans = ans.split('INST:')[0]
            ans = ans.strip()

            pans = ans.split('\n\n')
            if len(pans) == 1:
                ans = ans
            elif len(pans) == 2:
                if len(pans[0]) > len(pans[1]):
                    ans = pans[0]
                else:
                    ans = pans[1]
            elif len(pans) == 3:
                ans = ans # TBA
            
            ans = ans.strip()
            print (ans)
            ans1 = mdl.extract_questions(ans)
            print ('--------------->>>>')
            print (ans1)
            unique_list = list(set(ans1))
            ans1 = []
            black_lists = [
                "that this list",
                "entities",
                "that some of the",
                "hope this helps",
                "hope this information",
                "hope that",
                "the following",
            ]
            for item in unique_list:
                found = False
                for a_item in black_lists:
                    if a_item in item:
                        found = True
                        break
                if not found:
                    if len(item) < 100:
                        ans1.append(item)
            #print(ans1)
            ans1 = get_lemmatized_components(ans1)
            all_answers.extend(ans1)
            prec = calculate_precision(
                cand_list=ans1,
                gold_list=gtruth,
                validation_model=val_model,
                threshold=0.7,
            )
            rec = calculate_recall(
                cand_list=ans1,
                gold_list=gtruth,
                validation_model=val_model,
                threshold=0.7,
            )
            final_results.extend([dclm[i], prec, rec])
            print (dclm[i], prec, rec)

    # now all union
    unique_list = get_lemmatized_components(all_answers)
    prec = calculate_precision(
        cand_list=unique_list,
        gold_list=gtruth,
        validation_model=val_model,
        threshold=0.7,
    )

    rec = calculate_recall(
        cand_list=unique_list,
        gold_list=gtruth,
        validation_model=val_model,
        threshold=0.7,
    )
    final_results.extend([unique_list, prec, rec])
    print (final_results)
    return final_results


directory_paths = [
                   #("./experiment1_contextonly_withboundry/mixtral/",5),
                   #("./experiment1_contextonly_withboundry/granite/",3),
                   ("./experiment1_contextonly_withboundry/lamma/",1)]


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
        f"approach2_autoQ_context_guided_pipeline_generated_result_{model_initial}_{directory_path}.csv",
        index=False,
    )