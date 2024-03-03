import torch
import lightning
from minicons import cwe
import pandas as pd

from model import FFNModule, FeatureNormPredictor, FFNParams, TrainingParams

def fire():
    fires =  ['metaphor', 'destructive', 'artillery', 'hearth', 'compound']
    fire_data = [
        [
            ("An occasional shaft of sunlight penetrated the foliage and lit up the bronze trunks of the pines, touching them with fire.", "fire"),
            ("Adam swiftly read the titles, most of which contained romantic words like 'love', 'heart', 'arrow', 'passionate', 'fire', 'dream', 'kiss', and 'enchanted'.", "fire"),
            ("Changez said nothing, but shuffled backwards, away from the fire of Anwar's blazing contempt, which was fuelled by bottomless disappointment.", "fire"),
            ("Never again, except in the nostalgic hopefulness of a few—would the ceremonies be performed; gone were the offerings, the blood-shedding, the fire and incense, the gorgeous (and the plain) robes", "fire"),
        ],
        [
        ("There was a fire at Mr's store and they called it arson.", "fire"),
        ("An electrical short circuit started the fire, they think.", "fire"),
        ("Mr Green hopes the fire will provoke further pledges of aid to People In Need at 1113 Maryhill Road, Glasgow.", "fire"),
        ("A woman and seven children left a house in nearby Gudmunsen Avenue after a fire was discovered in a bedroom at around 7.30am on Saturday.", "fire"),
        ],
        [
        ("Small-arms fire scorched a web of gaps through the foam.", "fire"),
        ("Almost immediately, there was a brief burst of machinegun fire, which destroyed the three remaining wheels.", "fire"),
        ("'The hierarchy are now under fire because of the team's performance and are seeking to deflect criticism by blaming me", "fire"),
        ("Following an exchange of fire the Ju88 flew back to Sicily in a damaged condition and with one crewman wounded, but one Beaufighter—T3239 ‘B’ crewed by Flt.Lt.", "fire"),
        ],
        [
        ("or reading in the shadow of a fire;", "fire"),
        ("They all went over to the fire for plates of meat and bread.", "fire"),
        ("The light from the fire bathed her in a warm flickering glow as he lay down beside her.", "fire"),
        ("The bar is warm and cosy, with an open fire and oak beams", "fire"),
        ],
        [
        ("Now add the top of the fireback, bedding it on top of the lower half with a layer of fire cement", "fire"),
        ("That’s when the fire brigade arrived.", "fire"),
        ("Mr Small said fire alarms were installed and special voice tapes would tell people to leave the premises.", "fire"),
        ("A fire station is to be put up for sale, a council report has revealed.", "fire"),
        ]
    ]
    
    mod_names = ['binder', 'mcrae', 'buchanan']
    models = []
    for name in mod_names:
        models.append(FeatureNormPredictor.load_from_checkpoint(
        checkpoint_path='saved_models/chronis_et_al/bert_to_'+name+'_layer8_opt_prune_relu.ckpt',
        map_location=None
        ))

    labels = []
    for name in mod_names:
        with open ('saved_models/chronis_et_al/bert_to_'+name+'_layer8_opt_prune_relu.txt', "r") as file:
            labels.append([line.rstrip() for line in file.readlines()])

    lm = cwe.CWE('bert-base-uncased')
    for i in range(len(models)):
        model = models[i]
        for j in range(len(fire_data)):
            data = fire_data[j]
            embs = lm.extract_representation(data, layer=8)
            avg = embs.sum(0)/len(data)
            pred = model(avg)
            squeezed = pred.squeeze(0)
            df = pd.Series(squeezed.detach().numpy(), index = labels[i])
            df.sort_values(ascending=False, inplace=True)
            df.to_csv('results/'+mod_names[i]+'_fire_'+fires[j]+'.csv')

if __name__ == "__main__":
    fire()

            
    

    
    
    