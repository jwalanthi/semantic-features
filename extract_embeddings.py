from minicons import cwe
import torch
import numpy as np
import os
import pandas as pd
import gc

import re
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

from collections import defaultdict

def _find_word_form(word, sentence):
  '''find how word occurs in sentence'''
  matches = re.finditer(word, sentence.lower())
  for match in matches:
    span = match.span()
    if (span[0] == 0 or sentence[span[0]-1] == ' ') and (span[1] == len(sentence) or sentence[span[1]] == ' '):
      break
  return tuple(span)

def extract(model_name :str, token_file :str, gpu: int):
    lm = cwe.CWE(model_name, f'cuda:{gpu}')
    # initialize a dictionary of dictionaries
    word_embeddings = defaultdict(dict)
    print("Extracting embeddings for model "+ model_name+" with data in "+token_file)
    for root, dirs, files in os.walk(token_file):
        files.sort()
        for name in files:
            with open(os.path.join(root, name), "r") as f:
                reader = csv.reader(f, delimiter="\t")

                # save all queries separately
                # (needed because some words do not occur in
                # sentences in the same form and must be fixed first)
                queries = []
                for line in reader:
                    word, sentence, pos, id = line

                    # get the word's span
                    wordspan = _find_word_form(word, sentence)
                    # kick out sentences that are too long for the model
                    if len(sentence) <= lm.tokenizer.model_max_length:
                        queries.append((sentence, torch.tensor(wordspan)))
                # standardize the way words are identified
                WORD = queries[0][0][queries[0][1][0]:queries[0][1][1]].lower()

                # cool trick to initialize a dictionary with 0s for any new entry
                # so calling `layerwise_embeddings[0] += extracted embs will create a
                # new entry for '0' and add the extracted embs to it! cool right?
                layerwise_embeddings = defaultdict(lambda : torch.zeros(lm.dimensions).unsqueeze(0))

                dl = DataLoader(queries, batch_size=16)
                # tqdm is progress bar
                for batch in tqdm(dl):
                    sentences, words = batch
                    batched_query = list(zip(sentences, words))
                    layer_embs = lm.extract_representation(batched_query, layer='all')
                    for layer, embs in enumerate(layer_embs):
                        if layer == 0:
                            for i in range(0, embs.size(0)):
                                emb = embs[i]
                                query = batched_query[i]
                                if emb.isnan().any():
                                    print("nan detected in extracted embeddings, offending query:")
                                    print(f"query: {query}")
                                    raise Exception("nan found in embedding")
                        layerwise_embeddings[layer] += (embs.sum(0)/len(queries))

                layerwise_embeddings = dict(layerwise_embeddings)
                word_embeddings[WORD] = layerwise_embeddings
    return word_embeddings

def save(word_embeddings, model_name, save_dir):
    # what we can do now is save <layer> number of matrices
    vocab = []
    layer_matrices = defaultdict(list)
    for word, layer_embs in word_embeddings.items():
        vocab.append(word)
        for layer, embs in layer_embs.items():
            layer_matrices[layer].append(embs)

    # this gives us a dict of layer: torch.tensor we need to save
    layer_matrices = {k: torch.cat(v) for k,v in layer_matrices.items()}
    model_dir = os.path.join(save_dir,model_name)
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    words_file = open(model_dir+'/words.txt','w')
    words_file.write("\n".join(vocab))
    words_file.close()
    for layer, tensor in layer_matrices.items():
        torch.save(tensor, model_dir+"/layer"+str(layer)+'.pt')