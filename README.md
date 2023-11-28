# Semantic Features

### About

### Usage

To extract and save emebddings from a new model, use `extract_embs_main.py`

`python3 extract_embs_main.py --model=bert-base-uncased --dataset=bnc/token_lists`

After you've extracted the embeddings, you can use `check_savings.py` to make sure that the saved embeddings are of the correct format

`python3 check_savings.py --model=bert-base-uncased --dataset=/bnctoken_lists`