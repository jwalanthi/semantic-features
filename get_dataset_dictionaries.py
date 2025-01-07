import pandas as pd
import torch
import os
from collections import defaultdict

def get_dict_pair(norm: str, norm_file: str, embedding_directory: str, layer: int, translated=True, normalized=False):
    # get specified norm set
    if norm == 'binder':
        all_ratings, feature_list = get_binder_norms(norm_file)
    elif norm == 'buchanan':
        all_ratings, feature_list = get_buchanan_norms(norm_file, translated=translated, normalized=normalized)
    elif norm == 'mcrae':
        all_ratings, feature_list = get_mcrae_norms(norm_file)
    else:
        raise ValueError('norm must be binder, buchanan, or mcrae')
    
    # want to take the intersection of the words that have ratings and those for which we have embeddings
    with open(os.path.join(embedding_directory, 'words.txt')) as words_file:
        all_emb_words = list([line.rstrip() for line in words_file])
    in_both = set(all_ratings.keys()).intersection(all_emb_words)
    all_embs_tensor = torch.load(os.path.join(embedding_directory, 'layer'+str(layer)+'.pt'))

    # now to return that intersection as 2 dictionaries
    ratings = {}
    embeddings = {}
    for i in range(len(all_emb_words)):
        emb_word = all_emb_words[i]
        if emb_word in in_both:
            embeddings[emb_word] = all_embs_tensor[i]
            ratings[emb_word] = all_ratings[emb_word]
    
    return embeddings, ratings, feature_list


def get_binder_norms(norm_file):
    all_ratings = {}
    # read in csv, making sure that na's are interpreted as NaN
    ratings_df = pd.read_csv(norm_file, na_values=['na'])
    # fill in 0 for na's
    ratings_df.fillna(value=0, inplace=True)
    feature_cols = ratings_df.iloc[:,5:70].columns
    # columns that used to have an na are still strings, so change them to floats
    ratings_df[feature_cols] = ratings_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    for _, row in ratings_df.iterrows():
        word = row[1]
        binder_ratings = row[5:70]
        all_ratings[word] = torch.tensor(binder_ratings)
    # now i have word:tensor for all the words that have feature norms
    return all_ratings, feature_cols.tolist()
    
def get_mcrae_norms(norm_file):
    ratings_df = pd.read_csv(norm_file)
    # get a list of the features
    feature_list = ratings_df['Feature'].unique().tolist()
    feature_list.sort()
    # will make it easier to create the dictionary we want to have a sort of inverted list
    feature_indexes = {feature_list[i]:i for i in range(len(feature_list))}
    all_ratings = defaultdict(lambda : torch.zeros(len(feature_list)))
    for _, row in ratings_df.iterrows():
        word = row['Concept']
        feature = row['Feature']
        value = row['Prod_Freq']
        all_ratings[word][feature_indexes[feature]] = value
    # now i have a k-hot encoding for each of the words in the feature set
    return all_ratings, feature_list

def get_buchanan_norms(norm_file, translated=True, normalized=False):
    ratings_df = pd.read_csv(norm_file)
    # get a list of the features
    name_col = 'translated' if translated else 'feature'
    freq_col = 'frequency_'+name_col if not normalized else 'normalized_'+name_col
    feature_list = ratings_df[name_col].unique().tolist()
    feature_list.sort()
    # an inverted list
    feature_indexes = {feature_list[i]:i for i in range(len(feature_list))}
    all_ratings = defaultdict(lambda : torch.zeros(len(feature_list)))
    for _, row in ratings_df.iterrows():
        word = row['cue']
        feature = row[name_col]
        value = row[freq_col]
        all_ratings[word][feature_indexes[feature]] = value
    
    # now i have a k-hot encoding for each of the words in the feature set
    return all_ratings, feature_list

            
# testing
if __name__ == '__main__':
    embeddings, ratings, feature_list = get_dict_pair('mcrae','feature-norms/mcrae/concepts_features-Table1.csv','/home/shared/semantic_features/saved_embeddings/bert-base-uncased', 10, translated=True)
    print((ratings.keys() == embeddings.keys()))
    test_word = 'airplane'
    for i in range(len(ratings[test_word])):
        if ratings[test_word][i] != 0:
            print(feature_list[i], ratings[test_word][i])


