# Semantic Features

## About

`semantic-features` is an extensible, easy-to-use library for performing semantic analysis on words in context.

## Usage

### Embedding Extraction
To extract and save emebddings from a new model, use `extract_embs_main.py`

example usage: `python3 extract_embs_main.py --model=bert-base-uncased --dataset=bnc/less_token_lists --save_dir=saved_embs_test --gpu=3`

Parameters:

| Parameter | Required? | Values | Note |
| --------- | --------- | ------ | ---- |
| model | required | Name of LLM from HuggingFace | Autoregressive LLMs not recommended |
| dataset | required | path to dataset | BNC-like formatting expected |
| save_dir | required | path to directory to save embeddings to | |
| gpu | optional | which cuda to use | highly recommended |

After extracting the embeddings, you can use `check_savings.py` to make sure that the saved embeddings are of the correct dimensions and format

Sample usage: `python3 check_savings.py --embs=saved_embs_test/bert-base-uncased --dataset=bnc/less_token_lists`

Parameters:

| Parameter | Required? | Values |
| --------- | --------- | ------ |
| embs | required | path to embeddings extracted using `extract_embs_main.py` |
| dataset | required | path to dataset used to extract embeddings |

### Model Training and Optimization

To train a feature norm predictor model, use `model.py`, such as:

```
python model.py --norm=mcrae --norm_file=feature-norms/binder/WordSet1_Ratings.csv --embedding_dir=saved_embeddings/bert-base-uncased --lm_layer=10 --num_layers=2 --hidden_size=128 --dropout=0.1 --save_dir=models --save_model_name=test_name
```

To enable optimization, use the `optimize` and `prune` arguments. If optimizing, can also use `gpu` to accelerate training.

If you'd like to train a classifier for each layer of the LM, you can use the `train_all_layers.sh` script by running `./train_all_layers.sh` and following the prompts. All hyperparameters will be the same for each model except for the layer of the LM.