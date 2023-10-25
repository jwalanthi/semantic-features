# just here for debugging, checks that the dimensions are correct for the saved model
import argparse
from minicons import cwe
import os
import numpy as np

def check_dimensions(model_name, data):
    lm = cwe.CWE(model_name)
    # set nominal parameters
    nominal_layers = lm.layers
    nominal_dimensions = lm.dimensions
    nominal_word_count = 0
    for root, dirs, files in os.walk(data):
        nominal_word_count = len(files)
    shape_for_layer = [0]*nominal_layers
    actual_word_count = 0
    actual_layer_count = 0
    for root, dirs, files in os.walk(os.path.join('saved_embeddings',model_name), topdown=False):
        for name in files:
            if 'words' not in name:
                # print(os.path.join(root, name))
                layer = int(name.split('layer')[1].split('.')[0])
                as_np = np.load(os.path.join(root,name), allow_pickle=True)
                shape_for_layer[layer-1] = as_np.shape
            else:
                with open(os.path.join(root,name)) as wordfile:
                    words = [line.rstrip() for line in wordfile.readlines()]
                    actual_word_count = len(words)
    # check time
    if nominal_word_count == actual_word_count:
        print("Correct word count present in words.txt")
    for layer in range(len(shape_for_layer)):
        shape = shape_for_layer[layer]
        if shape != 0:
            actual_layer_count += 1
            if shape[0] != nominal_word_count:
                print('Incorrect word count for layer '+str(layer+1)+
                ' Expected: '+str(nominal_word_count)+' Actual: '+str(shape[0]))
            if shape[1] != nominal_dimensions:
                print('Incorrect dinmensions for layer '+str(layer+1))
    if nominal_layers == actual_layer_count:
        print("Correct number of layers saved")

def _parse_args():
    parser = argparse.ArgumentParser(description='extract_embs_main.py')

    parser.add_argument('--model', type=str, default='bert-base-uncased', help='specify model to extract from, default bert-base-uncased')
    parser.add_argument('--dataset', type=str, default='bnc/less_token_lists', help='file containting token csvs, default subset of bnc')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = _parse_args()
    args = vars(args)

    check_dimensions(args['model'], args['dataset'])
