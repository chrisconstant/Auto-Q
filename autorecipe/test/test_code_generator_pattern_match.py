import uuid
import json
import mlflow
from autorecipe.genai.GenAIChat import GenAIChatClient, extract_code
from dotenv import load_dotenv
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationReturnOptions,
)

load_dotenv()

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
api_key = "pak-KnqohFjgmJKou_eirLnIJMtbxdFzUYylnzScnLxVhY0"
api_endpoint = "https://bam-api.res.ibm.com"

CodeGenerationPromptTemplate = """You act as a Data Scientist and help to analyze a time series data. 
A time series data is a sequence of real value observations captured 
over regular time interval. The time dimension should be explicitily available in time series data. You will 
write a python code to detect a given temporal behavior in input time series. 
A time series can have several temporal behavior such as sudden increase, sudden drop, etc. We are only 
interested to detect such pattern toward end of the time series.

Your task is to generate a python code to detect the {temporalbehavior} temporal behavior toward the end of time series. 
There may be multiple ways of detecting the {temporalbehavior} temporal behavior, such as time series analysis 
approach (statsmodel based), statistical approach (moving average based) and machine learning approach (sklearn based). 
Do not generate explanation for solution. User will provide a 
sensor name that is attached to a particular asset or its component. 

We are interested in detecting sudden increase in {sensorname} sensor time series of {assetclass}'s 
{component} component using {approach}. The occurrence check should be focused toward the end part of the 
time series such as last 10% of the time series data. Do not provide example usage. Note that, if you do not
know solution using {approach}, then say I do not know.  

The name of python function should be detect_temporalbehavior with sensor_data as a first complulsory
argument. sensor_data is a panda dataframe that contains time as well as value. Other arguments 
should have default value. The python function should always return True or False.
There is only one function in the entire code and the entire code should be written inside a python inline 
block, i.e, it start from ```python and end by ```.
"""

params = {
    "decoding_method": DecodingMethod.GREEDY,
    "min_new_tokens": 200,
    "max_new_tokens": 2000,  # 1500,
    "stop_sequences": ["(TOKENSTOP)"],
    "return_options": TextGenerationReturnOptions(input_text=False, input_tokens=True),
    "moderations": ModerationParameters(
        hap=ModerationHAP(input=True, output=False, threshold=0.01)
    ),
}

with open("converted_dict.json") as f:
    samaps = json.load(f)

DisplayPromptTemplate = """We are interested in detecting sudden increase in {sensorname} sensor time series due to 
{failure_short} failure associated with {assetclass}'s {component} component using {approach}."""

final_answer = []

for component in samaps.keys():
    for failure in samaps[component].keys():
        for sensor in samaps[component][failure].keys():
            tem_behaviour = samaps[component][failure][sensor]["temporal behavior"]
            failure_short = failure.split("due to")[0]
            sensor = sensor.lower().split("sensor")[0]
            print(component, failure_short, sensor, tem_behaviour)

            for approach in [
                "statistical approach",
                "time series analysis approach",
                "machine learning approach",
            ]:
                dmessage = DisplayPromptTemplate.format(
                    temporalbehavior=tem_behaviour,
                    sensorname=sensor,
                    assetclass="Standby Generator",
                    component=component,
                    approach=approach,
                    failure_short=failure_short,
                )
                print(dmessage)

                CodeGenerationPrompt = CodeGenerationPromptTemplate.format(
                    temporalbehavior=tem_behaviour,
                    sensorname=sensor,
                    assetclass="Standby Generator",
                    component=component,
                    approach=approach,
                )

                print("--------------------------------------------")
                print(CodeGenerationPrompt)
                # Think: We now try to be very specific.

                LLMsets = [
                    "ibm-mistralai/mixtral-8x7b-instruct-v0-1-q",
                    "meta-llama/llama-2-70b-chat",
                ]

                creds = {
                    "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
                    "api_endpoint": "https://bam-api.res.ibm.com",
                }

                experiment_name = "MyExperiment_" + str(uuid.uuid4())
                experiment_id = mlflow.create_experiment(experiment_name)

                TestClient = GenAIChatClient(
                    name="DS",
                    description="this is test",
                    skill="data science",
                    model=LLMsets[0],
                    credentials=creds,
                    system_message=CodeGenerationPrompt,
                    stateful=True,
                    params=params,
                )

                ans = TestClient.create(
                    context=None,
                    messages=[{"content": "Generate only python code", "role": "user"}],
                    experiment_id=experiment_id,
                )
                # print(ans)
                pcode_ans = extract_code(ans)
                # print(pcode_ans)

                SynPrompt = """Generate python code with single function that produce synthetic time series for above code. 
                The name of function should be generate_synthetic_time_series with no compulsory
                argument. Do not use underscore sign in argument name. The default value is initialized such that if data is used as input to 
                detect_temporalbehavior, then the outcome is True. Do not include detect_temporalbehavior 
                in response. 
                """

                ans = TestClient.create(
                    context=None,
                    messages=[{"content": SynPrompt, "role": "user"}],
                    experiment_id=experiment_id,
                )
                # print(ans)
                scode_ans = extract_code(ans)
                print(pcode_ans[0][1])
                print(scode_ans[0][1])
                final_answer.append(
                    [
                        component,
                        failure_short,
                        sensor,
                        tem_behaviour,
                        dmessage,
                        pcode_ans[0][1],
                        scode_ans[0][1],
                        SynPrompt,
                        CodeGenerationPrompt,
                    ]
                )

print(final_answer)
import pandas as pd

A = pd.DataFrame(final_answer)
A.columns = [
    "component",
    "failure_short",
    "sensor",
    "tem_behaviour",
    "dmessage",
    "pcode_ans",
    "scode_ans",
    "SynPrompt",
    "CodeGenerationPrompt",
]

print (A)
A.to_csv('final_answer.csv',index=False)

"""
code_obj = compile(scode_ans[0][1], '<string>', 'exec')
exec(code_obj)

code_obj = compile(pcode_ans[0][1], '<string>', 'exec')
exec(code_obj)

# Call the function defined in the code string
result = generate_synthetic_time_series()
#print(result)
ad_score = detect_temporalbehavior(result)
print (ad_score)
"""

"""
local_vars = {}
exec(scode_ans[0][1],globals(), local_vars)
generate_synthetic_time_series = local_vars['generate_synthetic_time_series']
result = generate_synthetic_time_series()
print (result)
"""
