#### Code - 

```
python data_process.py --in_seq_data sss --in_meta_data ../../failure_mode_classification_iso/metadata.json --out_directory experiment --model_path_or_name intfloat/e5-large-v2 --out_description2label description2label --neg_num 4
```

```
python main.py --data_cache_dir hf_data --train_data ./preprocess/experimentdata --train_group_size 2 --query_max_len 512 --passage_max_len 128 --max_example_num_per_dataset 5 --has_template False --model_name_or_path intfloat/e5-large-v2 --cache_dir hf_cache --sentence_pooling_method mean --normlized True --flash_attn_2_enabled False --torch_dtype auto --output_dir hf_output
```