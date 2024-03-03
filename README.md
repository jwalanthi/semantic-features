# Semantic Features

### About

### Usage

To extract and save emebddings from a new model, use `extract_embs_main.py`

`python3 extract_embs_main.py --model=bert-base-uncased --dataset=bnc/token_lists`

After you've extracted the embeddings, you can use `check_savings.py` to make sure that the saved embeddings are of the correct format

`python3 check_savings.py --model=bert-base-uncased --dataset=/bnctoken_lists`

To format the data for use in training, use `get_dict_pair(norm: str, embedding_directory: str, layer: int, translated=True)` function from `get_dataset_dictionaries.py`

 - norm(str): must be `binder`, `buchanan`, or `mcrae`. Specifies which set of feature norms to use
 - embedding_directory(str): directory where lm embeddings for all layers are stored
 - layer(int): which layer of embeddings to use
 - translated(bool): optional, default `True`. Specifies whether to use raw or translated values for Buchanan norms. Ignored for other norms

To train a feature norm predictor model, use `model.py`, such as:

```
python model.py --norm=mcrae --embedding_dir=/home/shared/semantic_features/saved_embeddings/bert-base-uncased --lm_layer=10 --num_layers=2 --hidden_size=128 --dropout=0.1 --save_dir=/home/rmj2433/models --save_model_name=test_name
```