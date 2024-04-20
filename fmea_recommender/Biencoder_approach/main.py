from src.argument import (
    DataArguments,
    ModelArguments,
    RetrieverTrainingArguments as TrainingArguments,
)
import logging
from transformers import AutoConfig, AutoTokenizer
from src.data import TrainDatasetForEmbedding

logger = logging.getLogger(__name__)
from transformers import (
    HfArgumentParser,
    set_seed,
)

def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    data_args: DataArguments
    model_args: ModelArguments
    training_args: TrainingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
        padding_side='left' if "Llama" in model_args.model_name_or_path else 'right',
        truncation_side='right',
    )

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
    query, passages = train_dataset[2]
    print (query, passages)

if __name__ == "__main__":
    main()
