import torch
import lightning
from minicons import cwe
import pandas as pd

from model import FFNModule, FeatureNormPredictor, FFNParams, TrainingParams

def fire(mod_names: list):
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
            ("That's when the fire brigade arrived.", "fire"),
            ("Mr Small said fire alarms were installed and special voice tapes would tell people to leave the premises.", "fire"),
            ("A fire station is to be put up for sale, a council report has revealed.", "fire"),
        ]
    ]
    models = []
    for name in mod_names:
        models.append(FeatureNormPredictor.load_from_checkpoint(
        checkpoint_path=name+'.ckpt',
        map_location=None
        ))

    labels = []
    for name in mod_names:
        with open (name+'.txt', "r") as file:
            labels.append([line.rstrip() for line in file.readlines()])

    lm = cwe.CWE('bert-base-uncased')
    for i in range(len(models)):
        name = mod_names[i].split('/')[-1]
        model = models[i]
        model.eval()
        print(name+" hyperparams: ")
        print("    ",model.ffn_params)
        print("    ",model.training_params)
        
        for j in range(len(fire_data)):
            data = fire_data[j]
            embs = lm.extract_representation(data, layer=8)
            avg = embs.sum(0)/len(data)
            pred = torch.nn.functional.relu(model(avg))
            squeezed = pred.squeeze(0)
            df = pd.Series(squeezed.detach().numpy(), index = labels[i])
            df.sort_values(ascending=False, inplace=True)
            df.to_csv('results/fire/'+name+'_fire_'+fires[j]+'.csv')

def aann(buchanan_model: str):
    model_name = buchanan_model.split('/')[-1]
    words = ["days", "meals"]
    default_data = [
        "The family spent three lovely days in London",
        "They consumed five ugly meals",
    ]
    aann_data = [
        "The family spent a lovely three days in London",
        "They consumed an ugly five meals."
    ]
    
    lm = cwe.CWE('bert-base-uncased')
    model = FeatureNormPredictor.load_from_checkpoint(
        checkpoint_path=buchanan_model+'.ckpt',
        map_location=None
        )
    model.eval()
    print(model_name+" hyperparams: ")
    print("    ",model.ffn_params)
    print("    ",model.training_params)
    
    aann_features = ['measure', 'unit', 'one']
    with open (buchanan_model+'.txt', "r") as file:
            labels = [line.rstrip() for line in file.readlines()]
    for i in range(len(words)):
        word = words[i]
        def_embs = lm.extract_representation((default_data[i], word))
        def_pred = torch.nn.functional.relu(model(def_embs))
        def_pred = def_pred.squeeze(0)
        aann_embs = lm.extract_representation((aann_data[i], word))
        aann_pred = torch.nn.functional.relu(model(aann_embs))
        aann_pred = aann_pred.squeeze(0)
        df = pd.DataFrame({'feature': labels, 'default': def_pred.detach().numpy(), 'aann': aann_pred.detach().numpy()})
        df['default - aann'] = df['default'] - df['aann']
        df = df.sort_values('default - aann', ascending=False)
        aann_vals = df.loc[df['feature'].isin(aann_features)]
        df.to_csv('results/aann/'+model_name+'_'+word+'.csv')
        aann_vals.to_csv('results/aann/'+model_name+'_'+word+'_just3.csv')


def roles(binder_model: str):
    model_name = binder_model.split('/')[-1]
    words = [("dog", "cat"), ("chef", "onion")]
    natural = [
        "The dog chased the cat.",
        "The chef chopped the onion",
    ]
    swapped = [
        "The cat chased the dog", 
        "They onion chopped the chef"
    ]
    
    lm = cwe.CWE('bert-base-uncased')
    model = FeatureNormPredictor.load_from_checkpoint(
        checkpoint_path=binder_model+'.ckpt',
        map_location=None
        )
    model.eval()
    print(model_name+" hyperparams: ")
    print("    ",model.ffn_params)
    print("    ",model.training_params)
    with open (binder_model+'.txt', "r") as file:
        labels = [line.rstrip() for line in file.readlines()]
    # for each sentence
    for i in range(len(natural)):
        # for each word (subject and object)
        for j in range(2):
            word = words[i][j]
            nat = (natural[i], word)
            swap = (swapped[i], word)
            nat_embs = lm.extract_representation(nat)
            nat_pred = torch.nn.functional.relu(model(nat_embs))
            nat_pred = nat_pred.squeeze(0)
            swap_embs = lm.extract_representation(swap)
            swap_pred = torch.nn.functional.relu(model(swap_embs))
            swap_pred = swap_pred.squeeze(0)
            print
            df = pd.DataFrame({'feature': labels, 'natural': nat_pred.detach().numpy(), 'swapped': swap_pred.detach().numpy()})
            df['natural - swapped'] = df['natural'] - df['swapped']
            pos = df.sort_values('natural - swapped', ascending=False)
            neg = df.sort_values('natural - swapped', ascending=True)
            pos.to_csv('results/roles/'+model_name+'_'+word+'_descending.csv')
            neg.to_csv('results/roles/'+model_name+'_'+word+'_ascending.csv')



        

if __name__ == "__main__":
    mod_names = [
         'saved_models/chronis_et_al/bert_to_binder_layer8_stopped_opt',
         'saved_models/chronis_et_al/bert_to_buchanan_layer8_stopped_opt_long_enough',
         'saved_models/chronis_et_al/bert_to_mcrae_layer8_stopped_opt_long_enough',
         'saved_models/chronis_et_al/bert_to_buchanan_layer8_stopped_opt',
         'saved_models/chronis_et_al/bert_to_mcrae_layer8_stopped_opt']
    fire([mod_names[-1]])
    # aann(mod_names[1])
    # aann(mod_names[3])
    # roles(mod_names[0])

            
    

    
    
    