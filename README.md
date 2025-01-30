# Semantic Features

## About

`semantic-features` is an extensible, easy-to-use library for performing semantic analysis on words in context.

## Usage

### Embedding Extraction
To extract and save embeddings from a new model, use `extract_embs_main.py`, such as:

```
python3 extract_embs_main.py --model=bert-base-uncased --dataset=bnc/less_token_lists --save_dir=saved_embs_test --gpu=3
```

Parameters:

| Parameter | Required? | Value(s) | Note |
| --------- | --------- | ------ | ---- |
| model | required | str, name of LM from HuggingFace | autoregressive LLMs not recommended |
| dataset | required | str, path to dataset | BNC-like formatting and directory structure expected |
| save_dir | required | str, path to directory to save embeddings to | |
| gpu | optional | int, which cuda to use | highly recommended |

After extracting the embeddings, you can use `check_savings.py` to make sure that the saved embeddings are of the correct dimensions and format:

```
python3 check_savings.py --embs=saved_embs_test/bert-base-uncased --dataset=bnc/less_token_lists
```

Parameters:

| Parameter | Required? | Values |
| --------- | --------- | ------ |
| embs | required | str, path to embeddings extracted using `extract_embs_main.py` |
| dataset | required | str, path to dataset used to extract embeddings |

### Model Training and Optimization

To train a feature norm predictor model, use `model.py`, such as:

```
python model.py --norm=mcrae --norm_file=feature-norms/binder/WordSet1_Ratings.csv --embedding_dir=saved_embeddings/bert-base-uncased --lm_layer=10 --num_layers=2 --hidden_size=128 --dropout=0.1 --save_dir=models --save_model_name=test_name
```

To enable optimization, use the `optimize` and `prune` arguments. If optimizing, can also use `gpu` to accelerate training:

```
python model.py --norm=buchanan --norm_file=feature-norms/buchanan/cue_feature_words.csv --embedding_dir=saved_embeddings/albert-xxlarge-v2 --save_dir=saved_models/new/albert_models_all --optimize --gpu=2 --num_layers=2 --dropout=0.5 --num_epochs=100 --weight_decay=0 --early_stopping=6 --lm_layer=5 --save_model_name=albert_to_buchanan_layer5 --prune --normalized_buchanan 
```

Parameters:

| Parameter | Required? | Values | Note |
| --------- | --------- | ------ | ---- |
| norm | required | "binder", "buchanan", or "mcrae" | |
| norm_file | required | str, path to csv containing norm data | |
| embedding_dir | required | str, path to directory in which `extract_embs_main.py` saved embeddings | |
| lm_layer | required | int, layer of embeddings to use as source | |
| save_model_name | required | str, desired name of MLP | if optimizing, only best model saved |
| save_dir | required | str, path to save MLP to | |
| optimize | optional | present/not present | if present, hyperparameter tuning used |
| prune | optional | present/not present | only used if `--optimize` flag present, prunes unpromising trials |
| gpu | optional | int, which cuda to use | only used if `--optimize` flag present |
| num_layers | optional | int, default=2 | number of layers in MLP |
| hidden_size | optional | int, default=100 | hidden size of MLP, ignored if optimizing |
| dropout | optional | float, default=0.1 | dropout rate of MLP |
| num_epochs | optional | int, default=10 | number of epochs to train for, maximum if using early stopping |
| batch_size | optional | int, default=32 | batch size, ignored if optimizing |
| learning_rate | optional | float, default=0.001 | learning rate, ignored if optimizing |
| weight_decay | optional | float, default=0.0 | weight decay |
| early_stopping | optional | int, no default | number of epochs without progress to wait before early stopping |
| raw_buchanan | optional | present/not present | if present, indicates not to use translated values for Buchanan norms, ignored for other norms |
| normal_buchanan | optional | present/not present | if present, indicates to use normalized values for Buchanan norms, ignored for other norms |

To train a classifier for each layer of the LM, use the `train_all_layers.sh` script by running `./train_all_layers.sh` and following the prompts. All hyperparameters will be the same for each model except for the layer of the LM.
