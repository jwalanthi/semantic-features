import torch
import lightning
from minicons import cwe
import pandas as pd

from model import FFNModule, FeatureNormPredictor, FFNParams, TrainingParams

def test(): 
    model = FeatureNormPredictor.load_from_checkpoint(
        checkpoint_path='saved_models/bert_to_binder_layer11_default_relu.ckpt',
        map_location=None
    )
    
    print("model hyperparameters: ")
    for key,value in model.hparams.items():
        print("    {}: {}".format(key, value))

    # get a sample embedding to test
    data = [
        ("colorless green ideas sleep furiously", "ideas")
    ]
    lm = cwe.CWE('bert-base-uncased')
    emb = lm.extract_representation(data, layer=11)
    predicted= model(emb)
    squeezed = predicted.squeeze(0)
    print(squeezed.shape)

    ratings_df = pd.read_csv('feature-norms/binder/WordSet1_Ratings.csv', na_values=['na'])
    # fill in 0 for na's
    ratings_df.fillna(value=0, inplace=True)
    feature_cols = ratings_df.iloc[:,5:70].columns
    for i in range(len(feature_cols)):
        print(feature_cols[i]," : ", squeezed[i].item())


if __name__ == '__main__':
    test()