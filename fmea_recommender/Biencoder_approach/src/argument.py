from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class DataArguments:
    train_data: str = field(default=None, metadata={"help": "Path to train data"})

    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the cached data datasets"},
    )

    train_group_size: int = field(
        default=8,
        metadata={
            "help": "1 positive and train_group_size-1 negative passages for each query"
        },
    )

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000,
        metadata={"help": "the max number of examples for each dataset"},
    )
    
    has_template: bool = field(
        default=False,
        metadata={"help": "whether the data has template, used only for LLM"},
    )
