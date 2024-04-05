import pandas as pd
import ray
import mlflow
import uuid
from autorecipe.genai.GenAIChat import GenAIChatClient

from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationReturnOptions,
)


creds = {
    "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
    "api_endpoint": "https://bam-api.res.ibm.com",
}
LLMsets = [
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    "ibm-mistralai/mixtral-8x7b-instruct-v01-q",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-labrador-rc",
    "mistralai/mixtral-8x7b-instruct-v0-1",
]
params = {
    "decoding_method": DecodingMethod.GREEDY,
    "min_new_tokens": 100,
    "max_new_tokens": 3000,  # 1500,
    "stop_sequences": ["(TOKENSTOP)"],
    "return_options": TextGenerationReturnOptions(input_text=False, input_tokens=True),
    "moderations": ModerationParameters(
        hap=ModerationHAP(input=True, output=False, threshold=0.01)
    ),
}
model_id = 1

df_train = pd.read_csv("../fmea_result/autoQ_train_data_input_for_experiments.csv")
df_val = pd.read_csv("../fmea_result/autoQ_val_data_input_for_experiments.csv")
df_test = pd.read_csv("../fmea_result/autoQ_test_data_input_for_experiments.csv")

type_col = "TypeData.GenCompType"
long_desc_col = "long_description"
failure_loc_col = "failure_locations"

template = """
I have an industrial asset/equipments. Your task is to generate a redable one-line short 
description for the industrial equipment including the equipment/component type.

Asset: {asset}

"""


def get_prompts(df):
    """ """
    prompts = []
    for _, row in df.iterrows():
        asset = row[type_col]
        prompt = template.format(
            asset=asset
        )
        prompts.append(prompt)
    return prompts


train_prompts = get_prompts(df_train)
val_prompts = get_prompts(df_val)
test_prompts = get_prompts(df_test)

ray.init(num_cpus=8)


@ray.remote
def get_summaries(prompt):

    experiment_name = "MyExperiment_" + str(uuid.uuid4())
    experiment_id = mlflow.create_experiment(experiment_name)

    tmpClient = GenAIChatClient(
        name="Testing",
        description="I am testing classifier",
        skill="Generic",
        model=LLMsets[model_id],
        params=params,
        credentials=creds,
        system_message=prompt,
        stateful=True,
        stream=True,
    )

    answer = tmpClient.create(
        context=None,
        messages=[
            {
                "content": "Here is the description:\n",
                "role": "user",
            }
        ],
        experiment_id=experiment_id,
    )

    # "Did i miss any failure locations which is not captured in the Failure Locations?"
    # "can you improve your answer by grouping the failure locations into some category?"
    # "can you generate a redable summary of the above answer?"
    # "can you provide any thinking why these are the failure location?"
    # "generate list of failure locations"

    print (answer.strip())
    return answer.strip()


for model_id in [2, 1, 3]:
    model_initial = LLMsets[model_id].split("/")[1].split("-")[0]

    remote_call = []
    for pt in train_prompts:
        remote_call.append(get_summaries.remote(pt))
    train_summaries = ray.get(remote_call)
    df_train["shortsummary"] = train_summaries
    df_train.to_csv("short_summary_train_" + model_initial + ".csv")

    remote_call = []
    for pt in val_prompts:
        remote_call.append(get_summaries.remote(pt))
    val_summaries = ray.get(remote_call)
    df_val["shortsummary"] = val_summaries
    df_val.to_csv("short_summary_val_" + model_initial + ".csv")

    remote_call = []
    for pt in test_prompts:
        remote_call.append(get_summaries.remote(pt))
    test_summaries = ray.get(remote_call)
    df_test["shortsummary"] = test_summaries
    df_test.to_csv("short_summary_test_" + model_initial + ".csv")
