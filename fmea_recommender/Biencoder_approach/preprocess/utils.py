import json

# keywords : label, description, examples,


# this function read metadata.json to prepare the mapping
def get_item_text(infile_metadata, save_item_prompt_path=None):

    itemid2label = []
    itemid2description = []
    itemid2examples = []
    itemid2text = []

    for _, line in enumerate(open(infile_metadata)):
        line = json.loads(line)

        ## TitleName is Xbox data format; app_name and title are Steam data format
        title = line["label"]
        description = line["description"]
        examples = line["examples"]

        itemid2label.append(title)
        itemid2description.append(description)
        itemid2examples.append(examples)
        itemid2text.append(title)

    if save_item_prompt_path != None:
        with open(save_item_prompt_path, "w", encoding="utf-8") as f:
            for id, prompt in enumerate(itemid2text):
                line = {
                    "id": id,
                    "text": prompt,
                    "label": itemid2label[id],
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    return itemid2text, itemid2label, itemid2description, itemid2examples
