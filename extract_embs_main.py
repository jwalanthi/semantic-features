import argparse
from extract_embeddings import *
def _parse_args():
    parser = argparse.ArgumentParser(description='extract_embs_main.py')

    parser.add_argument('--model', type=str, default='bert-base-uncased', help='specify model to extract from, default bert-base-uncased')
    parser.add_argument('--dataset', type=str, default='bnc/less_token_lists', help='file containting token csvs, default subset of bnc')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = _parse_args()
    args = vars(args)

    word_embeddings = extract(args['model'], args['dataset'])
    save(word_embeddings, args['model'])
