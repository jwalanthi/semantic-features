# just here for debugging, checks that the dimensions are correct for the saved model
import argparse
from minicons import cwe
import os
import torch

def check_dimensions(model_path, data):
    model_name = model_path.split("/")[-1]
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
    for root, dirs, files in os.walk(model_path, topdown=False):
        for name in files:
            if 'pt' in name:
                # print(os.path.join(root, name))
                layer = int(name.split('layer')[1].split('.')[0])
                as_pt = torch.load(os.path.join(root,name))
                has_nan = as_pt.isnan().any()
                if has_nan: print("Layer {} has a nan".format(layer))
                shape_for_layer[layer-1] = as_pt.shape
            elif 'words' in name:
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
    parser = argparse.ArgumentParser(description='check_savings.py')

    parser.add_argument('--embs', type=str, required=True, help='specify path to embeddings')
    parser.add_argument('--dataset', type=str, required=True, help='specify path to data')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = _parse_args()
    args = vars(args)

    check_dimensions(args['embs'], args['dataset'])
