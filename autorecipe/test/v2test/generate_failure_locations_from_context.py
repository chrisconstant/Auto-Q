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


LLMsets = [
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v01-q",
    "mistralai/mixtral-8x7b-instruct-v0-1",
    "ibm/granite-13b-chat-v2",
]

model_id = 3
DEFAULT_CONFIG = {
    "model": LLMsets[model_id],
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

ray.init()

@ray.remote
def get_evaluation(asset_class, asset_class_result_file, gtruth):
    df = pd.read_csv(asset_class_result_file)
    val_model = SentenceTransformer("all-mpnet-base-v2")
    A = df.to_numpy()

    experiment_name = "MyExperiment_" + asset_class + "_" + str(uuid.uuid4())
    experiment_id = mlflow.create_experiment(experiment_name)
    stateful = True

    all_answers = []
    final_results = [asset_class]
    for i in range(A.shape[1]):
        SystemPrompt = SystemPromptTemplate.format(
            asset_class=asset_class, asset_description=A[0, i]
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
        )

        for pt in promptSeq:
            fpt = pt.format(asset_class=asset_class)
            ans = mdl.create(
                context=None,
                messages=[{"content": fpt, "role": "user"}],
                experiment_id=experiment_id,
            )

        ans1 = mdl.extract_questions(ans)
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
        all_answers.extend(ans1)

        prec = calculate_precision(
            cand_list=ans1,
            gold_list=gtruth,
            validation_model=val_model,
            threshold=0.7,
        )
        final_results.append(prec)

        rec = calculate_recall(
            cand_list=ans1,
            gold_list=gtruth,
            validation_model=val_model,
            threshold=0.7,
        )
        final_results.append(rec)

    # now all union
    unique_list = list(set(all_answers))
    prec = calculate_precision(
        cand_list=unique_list,
        gold_list=gtruth,
        validation_model=val_model,
        threshold=0.7,
    )
    final_results.append(prec)

    rec = calculate_recall(
        cand_list=unique_list,
        gold_list=gtruth,
        validation_model=val_model,
        threshold=0.7,
    )
    final_results.append(rec)
    print (final_results)
    return final_results

directory_path = "./experiment1_contextonly_withboundry/"
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
        gtruth = extract_things_from_string(goldenset[item_index])
        print (item)
        refs.append(get_evaluation.remote(item, filename, gtruth))

parallel_returns = ray.get(refs)

res = pd.DataFrame(parallel_returns)
model_initial = LLMsets[model_id].split("/")[1].split("-")[0]
directory_path = directory_path.replace('/','').replace('.','')
res.to_csv(f'step1_generated_result_{model_initial}_{directory_path}.csv',index=False)

