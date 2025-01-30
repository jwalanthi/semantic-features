import argparse
from extract_embeddings import *
def _parse_args():
    parser = argparse.ArgumentParser(description='extract_embs_main.py')

    parser.add_argument('--model', type=str, required=True, help='specify model to extract from, default bert-base-uncased')
    parser.add_argument('--dataset', type=str, required=True, help='file containting token csvs, default subset of bnc')
    parser.add_argument("--save_dir", type=str, required=True, help="directory to save embeddings to")
    parser.add_argument("--gpu", type=int, default=None, help="if using gpu, which device (recommended)")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = _parse_args()
    args = vars(args)

    word_embeddings = extract(args['model'], args['dataset'], args['gpu'])
    save(word_embeddings, args['model'], args['save_dir'])
