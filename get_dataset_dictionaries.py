import pandas as pd
import torch
import os
from collections import defaultdict

def get_dict_pair(norm: str, embedding_directory: str, layer: int, translated=True):
    if norm == 'binder':
        all_ratings = get_binder_norms()
        feature_list = []
    elif norm == 'buchannan':
        all_ratings, feature_list = get_buchannan_norms(translated=translated)
    elif norm == 'mcrae':
        all_ratings, feature_list = get_mcrae_norms()
    else:
        raise ValueError('norm must be binder, buchannan, or mcrae')
    
    # want to take the intersection of the words that have ratings and those for which we have embeddings
    with open(os.path.join(embedding_directory, 'words.txt')) as words_file:
        all_emb_words = list([line.rstrip() for line in words_file])
    
    in_both = set(all_ratings.keys()).intersection(all_emb_words)
    all_embs_tensor = torch.load(os.path.join(embedding_directory, 'layer'+str(layer)+'.pt'))
    ratings = {}
    embeddings = {}
    for i in range(len(all_emb_words)):
        emb_word = all_emb_words[i]
        if emb_word in in_both:
            embeddings[emb_word] = all_embs_tensor[i]
            ratings[emb_word] = all_ratings[emb_word]
    
    return embeddings, ratings, feature_list


def get_binder_norms():
    all_ratings = {}

    ratings_df = pd.read_csv('feature-norms/binder/WordSet1_Ratings.csv', na_values=['na'])
    # fill in 0 for na's
    ratings_df.fillna(value=0, inplace=True)
    feature_cols = ratings_df.iloc[:,5:70].columns
    ratings_df[feature_cols] = ratings_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    for _, row in ratings_df.iterrows():
        word = row[1]
        all_ratings[word] = torch.tensor(row[5:70])
    # now i have word:tensor for all the words that have feature norms
    return all_ratings
    
def get_mcrae_norms():
    ratings_df = pd.read_csv('feature-norms/mcrae/concepts_features-Table1.csv')
    # get a list of the features
    feature_list = ratings_df['Feature'].unique().tolist()
    feature_list.sort()
    feature_indexes = {feature_list[i]:i for i in range(len(feature_list))}
    all_ratings = defaultdict(lambda : torch.zeros(len(feature_list)))
    for _, row in ratings_df.iterrows():
        word = row[0]
        all_ratings[word][feature_indexes[row[1]]] = 1
    
    # now i have a one-hot encoding for each of the words in the feature set
    return all_ratings, feature_list

def get_buchannan_norms(translated=True):
    ratings_df = pd.read_csv('feature-norms/buchanan/cue_feature_words.csv')
    # get a list of the features
    column = 'translated' if translated else 'feature'
    feature_list = ratings_df[column].unique().tolist()
    feature_list.sort()
    # an inverted list
    feature_indexes = {feature_list[i]:i for i in range(len(feature_list))}
    all_ratings = defaultdict(lambda : torch.zeros(len(feature_list)))
    for _, row in ratings_df.iterrows():
        word = row[1]
        all_ratings[word][feature_indexes[row[column]]] = 1
    
    # now i have a one-hot encoding for each of the words in the feature set
    return all_ratings, feature_list
            
if __name__ == '__main__':
    embeddings, ratings, feature_list = get_dict_pair('mcrae', '/home/shared/semantic_features/saved_embeddings/bert-base-uncased', 10, translated=True)
    print((ratings.keys() == embeddings.keys()))
    print(embeddings['airplane'])

