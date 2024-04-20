import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple, Union
import torch

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
from .argument import DataArguments

class TrainDatasetForEmbedding(Dataset):
    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer):
        print (args.train_data)
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(args.train_data, file),
                    split="train",
                    cache_dir=args.data_cache_dir,
                )
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(
                            list(range(len(temp_dataset))),
                            args.max_example_num_per_dataset,
                        )
                    )
                column_names = temp_dataset.column_names
                remove_columns = ["user_id", "item_id", "neg_ids", "pos_id"]
                remove_columns = [c for c in remove_columns if c in column_names]
                temp_dataset = temp_dataset.remove_columns(remove_columns)
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)

        ## shuffle self.dataset
        self.dataset = self.dataset.shuffle()

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        query = self.dataset[item]["query"]
        passages = []
        pos = random.choice(self.dataset[item]["pos"])
        passages.append(pos)

        if len(self.dataset[item]["neg"]) < self.args.train_group_size - 1:
            num = math.ceil(
                (self.args.train_group_size - 1) / len(self.dataset[item]["neg"])
            )
            negs = random.sample(
                self.dataset[item]["neg"] * num, self.args.train_group_size - 1
            )
        else:
            negs = random.sample(
                self.dataset[item]["neg"], self.args.train_group_size - 1
            )
        passages.extend(negs)
        if self.args.has_template:
            query = f"query: {query}"
            for i in range(len(passages)):
                passages[i] = f"passage: {passages[i]}"

        return query, passages
