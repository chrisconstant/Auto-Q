import os
import argparse
from transformers import AutoTokenizer
from utils import get_item_text
import random
import json
from template import description2label_template

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument("--in_seq_data", type=str, help="")
    parser.add_argument("--in_meta_data", type=str, help="")
    parser.add_argument("--out_directory", type=str, help="")
    parser.add_argument("--model_path_or_name", type=str, help="")
    parser.add_argument("--out_description2label", type=str, help="")
    parser.add_argument("--neg_num", type=int, help="")

    args = parser.parse_args()
    return args

def gen_description2label(itemid2label, itemid2description, args):
    """_summary_

    :param itemid2label: _description_
    :type itemid2label: _type_
    :param itemid2description: _description_
    :type itemid2description: _type_
    :param args: _description_
    :type args: _type_
    """
    count = 0
    total_q_len = 0
    max_q_len = 0
    min_q_len = 100000

    with open(args.out_description2label, 'w') as f:
        
        for index in range(len(itemid2label)):

            if random.random() < 0.5:
                template = "{}"
            else:
                template = random.choice(description2label_template)

            query = ''

            query += itemid2description[index]
            template_length = len(tokenizer.tokenize(template))
            tokens = tokenizer.tokenize(query)[:args.max_seq_len-template_length]
            truncated_query = tokenizer.convert_tokens_to_string(tokens).strip().strip(',')
            query = template.format(truncated_query)

            q_len = template_length + len(tokens)
            total_q_len += q_len
            max_q_len = max(max_q_len, q_len)
            min_q_len = min(min_q_len, q_len)

            target_item = itemid2label[index]
            neg_items = []
            while len(neg_items) < args.neg_num:
                neg_item = random.randint(0, len(itemid2label)-1)
                if neg_item != index:
                    neg_items.append(itemid2label[neg_item])

            output = {
                'user_id': index,
                'item_id': target_item,
                'neg_ids': neg_items,
                'query': query,
                'pos': [target_item],
                'neg': neg_items
            }
            f.write(json.dumps(output) + '\n')
            count += 1

    print('total samples: ', count)
    print('avg q len: ', total_q_len/count)
    print('max q len: ', max_q_len)
    print('min q len: ', min_q_len)

if __name__ == "__main__":
    args = parse_args()
    print("Output directory path:", args.out_directory)

    try:
        if not os.path.exists(args.out_directory):
            os.makedirs(args.out_directory)
    except FileNotFoundError as e:
        print("Error:", e)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, use_fast=True)
    args.max_seq_len = tokenizer.model_max_length

    itemid2text, itemid2label, itemid2description, itemid2examples = get_item_text(
        args.in_meta_data
    )
    # print (itemid2text, itemid2label, itemid2description, itemid2examples)
    gen_description2label(itemid2label, itemid2description, args)


# python data_process.py --in_seq_data sss --in_meta_data ../../failure_mode_classification_iso/metadata.json --out_directory experiment --model_path_or_name intfloat/e5-large-v2 --out_description2label description2label --neg_num 4
# python data_process.py --in_seq_data sss --in_meta_data ../../failure_mode_classification_iso/metadata.json --out_directory experiment --model_path_or_name intfloat/e5-large-v2 --out_description2label description2label --neg_num 4