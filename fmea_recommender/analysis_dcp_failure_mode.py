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
We give you the asset and the locations where the asset failed. You need to select failure codes from the 
following lists that can cause the asset to fail at that location. Please also consider the 
component/sub-component of asset.

Here is a failure mode and its description:
1. Burned, Consumed/damaged/deformed because of overheating.
2. Cracked, Damaged and showing lines on the surface from having split without coming apart.
3. Deformed, Not having the normal or natural shape or form.
4. External leak, Accidentally lose or admit contents. Source of leak is visible.
5. Fractured, Fractured or damaged and no longer in one piece.
6. Failure to close, equipment does not close on demand.
7. Failure to open, equipment does not open on demand.
8. Failure to start/function on demand, the equipment does not start/function on demand.
9. Internal leak, Accidentally lose or admit contents. Source of leak is not visible.,
10. Lubrication/Fluid Sampling, Unsatisfactory fluid/lubrication sample.
11. Material build-up, Accumulation of  foreign/external material.
12. Noise, A sound that is loud and causes disturbance.
13. Parameter deviation, A parameter is not controlled as set.
14. Plugged, A constraint in flow conductor.
15. Power/signal failure, Power or signal loss in electrical system.
16. Rust, Build up of corrosion products.
17. Spurious Alarm, which is described as Unexpected/false alarm.
18. Slippage, Decrease of transmitted power in a mechanical system caused by slipping.
19. Does not stop on demand, equipment does not stop on demand.
20. Stuck, equipment does not move on demand.
21. Loss of Explosion Protection (EX) Integrity, Loss of Explosion Protection (EX) Integrity.
22. Thickness reduction, Reduction in thickness measurement / loss of material.
23. Fix Vibration, Vibration is higher that the established limit.

User has provided the following information: 

Asset: {asset}

Failure Locations: {failurelocation}

"""


def get_prompts(df):
    """ """
    prompts = []
    for _, row in df.iterrows():
        asset = row[type_col]
        failurelocation = row[failure_loc_col]
        prompt = template.format(
            asset=asset,
            failurelocation=failurelocation,
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
        stateful=False,
        stream=True,
    )

    answer = tmpClient.create(
        context=None,
        messages=[
            {
                "content": "Do not generate the additional information. Here is a failure mode for each location:\n",
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

    ans = answer.strip().lower()

    final_ans = ''
    string_to_check = ["Burned", "Cracked", "Deformed", "leak", "Fractured", 
                       "Failure to close", "Failure to open", "Failure to start", 
                       "function on demand", "Lubrication", "build-up", "Noise", 
                       "deviation", "Plugged", "Power/signal failure", "Rust", 
                       "Spurious Alarm", "Slippage", "Does not stop on demand", 
                       "Stuck", "Loss of Explosion", "Thickness reduction", "Vibration"]

    for item in string_to_check:
        if item.lower() in ans:
            final_ans = final_ans + ', ' + item

    print (final_ans)
    return final_ans

for model_id in [3]:
    model_initial = LLMsets[model_id].split("/")[1].split("-")[0]

    remote_call = []
    for pt in train_prompts:
        remote_call.append(get_summaries.remote(pt))
    train_summaries = ray.get(remote_call)
    df_train["failuremode"] = train_summaries
    df_train.to_csv("failuremode_train_" + model_initial + ".csv")

    remote_call = []
    for pt in val_prompts:
        remote_call.append(get_summaries.remote(pt))
    val_summaries = ray.get(remote_call)
    df_val["failuremode"] = val_summaries
    df_val.to_csv("failuremode_val_" + model_initial + ".csv")
    
    remote_call = []
    for pt in test_prompts:
        remote_call.append(get_summaries.remote(pt))
    test_summaries = ray.get(remote_call)
    df_test["failuremode"] = test_summaries
    df_test.to_csv("failuremode_test_" + model_initial + ".csv")
