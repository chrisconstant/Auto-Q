from sentence_transformers import SentenceTransformer
from performance_study import (
    setup_few_shot_retrieval,
    get_bam_config,
    get_related_examples,
)
from new_step1_ps import generate_components
from genai.credentials import Credentials
import new_step1_ps
import validation
import pandas as pd

api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_url = "https://bam-api.res.ibm.com"
creds = Credentials(api_key, api_endpoint=api_url)
val_model = SentenceTransformer("all-mpnet-base-v2")
perms = [(0, 1, 2)]

exp_conf = {
    "train_data": "all_ids_deduplicated_train_short_desc_to_failure_locations.csv",
    "val_data": "all_ids_deduplicated_val_short_desc_to_failure_locations.csv",
    "test_data": "all_ids_deduplicated_test_short_desc_to_failure_locations.csv",
    "context_val_data": "asset_desc_granitev2.csv",
    "max_num_shots": 3,  # Max number of shots or examples to include in the prompt (-1 equates to one random example).
    "decoding_method": "greedy",
    "max_new_tokens": 250,
    "repetition_penalty": 1.2,
    "description": "new prompt",
}

all_llm_models = [
    {
        "name": "ibm/granite-13b-chat-v2",
        "prompt_func": "build_labrador_prompt_dfs",
        "output_parser": "output_parser_llama2",
    },
    {
        "name" : "ibm/granite-13b-instruct-v2",
        "prompt_func" : "build_instruct_prompt_dfsp",
        "output_parser" : "output_parser_instruct"
    }
]


def build_cand_col_name(llm_model, prefix, suffix=None):
    """Construct candidate column name from llm_model and perm"""
    model_name = llm_model["name"].split("/")[-1]
    if suffix:
        cand_col_name = "autoqfls_" + model_name + "_" + str(suffix)
    else:
        cand_col_name = "autoqfls_" + model_name
    return prefix + cand_col_name


def process_model_perm(
    llm_model,
    step_1_data,
    creds,
    dynamic_shots_cfg,
    exp_conf,
    permutation=None,
    step_1_context_data=None,
):
    """Generate output for model 'llm_model' and calculate metrics."""
    prompt_func = getattr(new_step1_ps, llm_model["prompt_func"])
    output_parser = getattr(new_step1_ps, llm_model["output_parser"])
    model_name = llm_model["name"].split("/")[-1]
    print("Model:", model_name)
    print("Permutation:", permutation)
    print(prompt_func, output_parser)

    step_1_context_data_list = list(step_1_context_data['asset_dec'])


    cand_col_name = build_cand_col_name(llm_model, "dfsp_", permutation)

    step_1_data[cand_col_name] = ""

    bam_config = get_bam_config(
        credentials=creds,
        decoding_method=exp_conf["decoding_method"],
        max_new_tokens=exp_conf["max_new_tokens"],
        repetition_penalty=exp_conf["repetition_penalty"],
    )

    dfsp_fls, raw_responses = generate_components(
        step_1_data["short_descriptions"],
        dynamic_shots_cfg,
        llm_model["name"],
        bam_config,
        prompt_method=prompt_func,
        output_parser=output_parser,
        max_num_egs_in_prompt=exp_conf["max_num_shots"],
        permutation=permutation,
        step_1_context_data_list=step_1_context_data_list,
    )
    step_1_data[cand_col_name + "_raw"] = raw_responses
    step_1_data[cand_col_name] = dfsp_fls

    return step_1_data


def run_pipeline_on_split(split, val_model, context_data):
    step_1_data = pd.read_csv(exp_conf[split])
    step_1_context_data = None
    if context_data:
        step_1_context_data = pd.read_csv(exp_conf[context_data])
        step_1_context_data = step_1_context_data[["asset_dec", "asset_para"]]
    step_1_data.rename(
        columns={"failure_locations": "gold_failure_locations"}, inplace=True
    )
    step_1_data.rename(
        columns={"TypeData.GenCompType": "short_descriptions"}, inplace=True
    )

    for llm_model in all_llm_models:
        model_name = llm_model["name"].split("/")[-1]
        print("Model:", model_name)
        for perm in perms:
            cand_col_name = build_cand_col_name(llm_model, "dfsp_", perm)
            step_1_data = process_model_perm(
                llm_model,
                step_1_data,
                creds,
                dynamic_shots_cfg,
                exp_conf,
                perm,
                step_1_context_data,
            )
            step_1_data = validation.calculate_metrics(
                step_1_data, val_model, cand_col_name
            )
    return step_1_data


dynamic_shots_cfg = setup_few_shot_retrieval(
    train_data_path=exp_conf["train_data"], query_column="TypeData.GenCompType"
)
step_1_data_val = run_pipeline_on_split(
    "val_data", val_model, context_data="context_val_data"
)
step_1_data_val.to_csv("output.csv", index=False)
print(step_1_data_val)
