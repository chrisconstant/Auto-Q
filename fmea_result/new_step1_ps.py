import performance_study as ps
import ast
import re

def sort_components(components):
    list_components = ast.literal_eval(components)
    list_components.sort()
    return str(list_components)

def build_all_prompts(
    sds, best_k_matches, dynamic_shots_cfg, prompt_method, permutation, step_1_context_data_list
):
    """Build and gather all prompts into a list."""
    all_prompts = []
    for cnt, bm in enumerate(best_k_matches):
        gold_sds = []
        gold_flss = []
        for idx in bm: # Prepare relevant examples.
            gold_sd = dynamic_shots_cfg["train_data"].iloc[idx]["TypeData.GenCompType"]
            gold_fls = dynamic_shots_cfg["train_data"].iloc[idx]["failure_locations"]
            # Sort Examples!!!
            gold_fls = sort_components(gold_fls) # Sort examples alphabetically
            gold_sds.append(gold_sd)
            gold_flss.append(gold_fls)

        if permutation:
            gold_sds, gold_flss = ps.shuffle_examples(
                gold_sds, gold_flss, permutation
            )
        val_sd = sds[cnt]
        if step_1_context_data_list:
            prompt = prompt_method(gold_sds, gold_flss, val_sd, step_1_context_data_list[cnt])
        else:
            prompt = prompt_method(gold_sds, gold_flss, val_sd)
        all_prompts.append(prompt)
    
    import pandas as pd
    X = pd.DataFrame(all_prompts)
    X.columns = ['prompts']
    X.to_csv('prompt.csv',index=False)
    return all_prompts

def generate_components(
    sds,
    dynamic_shots_cfg,
    model_id,
    bam_config,
    prompt_method,
    output_parser,
    max_num_egs_in_prompt=4,
    permutation=None,
    step_1_context_data_list=None,
):
    """Input: sds - A list of long descriptions.
    Output: For each inputted long description, a list of failure locations."""
    best_k_matches = ps.get_related_examples(sds, dynamic_shots_cfg, max_num_egs_in_prompt)
    all_prompts = build_all_prompts(
        sds, best_k_matches, dynamic_shots_cfg, prompt_method, permutation, step_1_context_data_list
    )
    responses = list(
        bam_config['client'].text.generation.create(
            model_id=model_id,
            inputs=all_prompts,
            parameters=bam_config['generate_params']
        )
)
    all_raw_texts = []
    for r, prompt in zip(responses, all_prompts):
        #print("Prompt:\n", prompt)
        #print("Response:\n", r.results[0].generated_text)
        #print("----------------------------------")
        all_raw_texts.append(r.results[0].generated_text)

    all_failure_locations = []
    for raw_text in all_raw_texts:
        all_failure_locations.append(output_parser(raw_text))

    return all_failure_locations, all_raw_texts

#####################################################################################################
########################################## Output Parsers ###########################################
#####################################################################################################

def output_parser_instruct(raw_text):
    #print("op: raw_text:", raw_text)
    fls = raw_text.split("Output:")[-1]
    fls = fls.split("', '")
    fls = [fl.replace("'", "") for fl in fls]
    fls = [fl.replace("\n", "").replace("{", "").replace("}", "") for fl in fls]
    fls = [fl.replace("\n", "").replace("[", "").replace("]", "") for fl in fls]
    fls = [fl.strip(", ") for fl in fls]
    fls = list(set(fls))
    #print("op: fls:", fls)
    return fls

#####################################################################################################
########################################## Prompt Builders ##########################################
#####################################################################################################


def build_instruct_prompt_dfsp(gold_sds, gold_flss, val_sd, asset_context=None):
    """Build prompt for instruction tuned models."""

    # - Introduction
    prompt = """You are an expert in asset management. """
    prompt += """The identification of failure locations is essential to ensuring the good performance and availability of your equipment. """
    # -- Example prefix
    #prompt += "You can assume that similiar equipment have similar failure locations. "
    prompt += "The examples below show equipment and their failure locations.\n"

    # -- list(Example entry)
    for gold_sd, gold_fls in zip(gold_sds, gold_flss):
        prev_prompt = prompt
        prompt += "Input: {}.\n".format(gold_sd)
        prompt += "Output: {}.\n\n".format(gold_fls)
        if (
            len(prompt) > 8000
        ):  # KL: Stop prompt from getting too big...make this smarter...
            prompt = prev_prompt
    #- Instruction 
    prompt += "Now please complete this example:\n"
    prompt += "Input: {}.\n".format(val_sd)
    if asset_context:
        prompt += "Equipment Additional Description: {}\n".format(asset_context)
    prompt += "Output: "
    return prompt


def build_mixtral_dialogue_prompt_dfs(gold_sds, gold_lds, val_sd):
    # Step 1 DFSP Prompt for llama-chat
    prompt = "<s>[INST]\n"
    prompt += """You are an expert in industrial asset management.  """
    prompt += """The identification of failure locations is essential to ensuring the good performance and availability of your equipment. """
    prompt += """Please complete the final example in exactly the same format as the provided examples. """
    #prompt += """You can assume that similiar equipment types have similar failure locations."""
    prompt += """Please be concise, just provide the failure locations, don't provide any other information."""
    prompt += """Only provide answers that you are absolutely sure about."""
    prompt += "\n\n"

    for cnt, (gold_sd, gold_ld) in enumerate(zip(gold_sds, gold_lds)):
        prev_prompt = prompt
        if cnt == 0:
            prompt += "Generate failure locations for the following equipment.\n"
        if cnt > 1:
            prompt += "[INST]"
        prompt += "Equipment Description: {}.\n".format(gold_sd)
        prompt += "[/INST]\n"
        prompt += "Failure Locations:\n"
        prompt += "* {}\n</s>".format(gold_ld)
        if (
            len(prompt) > 9000
        ):  # KL: Stop prompt from getting too big...make this smarter...
            prompt = prev_prompt
            break
    prompt += "<s>[INST]Generate failure locations for the following equipment.\n"
    prompt += "Equipment Description: {}\n".format(val_sd)
    prompt += "[/INST]"

    return prompt


def build_labrador_prompt_dfs(gold_lds, gold_flss, val_ld, asset_context=None):
    """New Step 1 Prompt."""
    prompt = "<|system|>\n"
    prompt += """You are an expert in asset management. """
    prompt += (
        """The identification of failure locations is essential to ensuring the good """
    )
    prompt += """performance and availability of your equipment. """
    prompt += """Please complete the final example in exactly same format as the provided examples. """
    prompt += """Please be concise, just provide the failure locations, don't provide any other information."""
    prompt += """Only provide answers that you are absolutely sure about.\n"""

    prompt += "<|user|>"

    for cnt, (gold_ld, gold_fls) in enumerate(zip(gold_lds, gold_flss)):
        gold_fls_p = ast.literal_eval(gold_fls)
        prev_prompt = prompt
        if cnt > 0:
            prompt += (
                "\nQuestion: Generate failure locations for the following equipment.\n"
            )
        prompt += "Text: Equipment Description: {}.\n".format(gold_ld)
        prompt += "Answer: Sure! Here are some potential failure locations for that equipment:\n"
        for fl in gold_fls_p:
            prompt += "* {}\n".format(fl)
        if (
            len(prompt) > 8000
        ):  # KL: Stop prompt from getting too big...make this smarter...
            prompt = prev_prompt
            break
    prompt += "\n The examples represent important known failure locations for asset related to your task. If there's anything missing in the previous examples add it to the response.\n"
    prompt += "Question: Generate failure locations for the following equipment.\n"
    prompt += "Text: Equipment Description: {}\n".format(val_ld)
    if asset_context:
        prompt += "Equipment Additional Description: {}\n".format(asset_context)
    prompt += "Answer:\n<|assistant|>\n"

    return prompt

def output_parser_mixtral_dialogue(raw_text):
    """Output parser for mixtral_dialogue."""
    if ']' in raw_text:
        raw_text = raw_text[raw_text.find("["):raw_text.find("]")+1]
    else:
        raw_text = raw_text[raw_text.find("["):raw_text.rfind(",")] + ']'

    return output_parser_instruct(raw_text)

def output_parser_llama2(raw_text):
    """Output parser for llama2 chat model."""
    #print("raw_text:", raw_text)

    eg_lines = raw_text.splitlines()
    failure_locations = []
    patt = re.compile(r"^[0-9\*\. ]+")
    for l in eg_lines:
        l = l.strip()
        if re.search("^[0-9\*]", l):
            l = patt.sub("", l)
            l = l.replace("failure", "")
            l = l.strip(": ")
            failure_locations.append(l)
    #print("failure_locations:", failure_locations)

    return failure_locations


def build_llama2_chat_prompt_dfsp(gold_lds, gold_flss, val_ld):
    # Step 2 DFSP Prompt for llama-chat
    prompt = "[INST] <<SYS>>\n"
    prompt += """You are an expert in asset management. """
    prompt += """The identification of failure locations is essential to ensuring the good """
    prompt += """performance and availability of your equipment. """
    prompt += """Please complete the final example in exactly same format as the provided examples. """
    prompt += """Please be concise, just provide the failure locations, don't provide any other information."""
    prompt += """Only provide answers that you are absolutely sure about."""
    prompt += "<</SYS>>\n\n"

    for cnt, (gold_ld, gold_fls) in enumerate(zip(gold_lds, gold_flss)):
        gold_fls_p = ast.literal_eval(gold_fls)
        prev_prompt = prompt
        if cnt > 0:
            prompt += "[INST]]Generate failure locations for the following equipment.\n"
        prompt += "Equipment Description: {}.\n".format(gold_ld)
        prompt += "[/INST]\n"
        prompt += (
            "Sure! Here are some potential failure locations for that equipment:\n"
        )
        for fl in gold_fls_p:
            prompt += "* {}\n".format(fl)
        if (
            len(prompt) > 8000
        ):  # KL: Stop prompt from getting too big...make this smarter...
            prompt = prev_prompt
            break
    prompt += "[INST]Generate failure locations for the following equipment.\n"
    prompt += "Equipment Description: {}\n".format(val_ld)
    prompt += "[/INST]"

    return prompt
