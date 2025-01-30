import torch
import lightning
from torch.utils.data import Dataset
from typing import Any, Dict
import argparse
from pydantic import BaseModel
from get_dataset_dictionaries import get_dict_pair
import os
import shutil

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from functools import partial

class FFNModule(torch.nn.Module):
    """
    A pytorch module that regresses from a hidden state representation of a word
    to its continuous linguistic feature norm vector.

    It is a FFN with the general structure of:
    input -> (linear -> nonlinearity -> dropout) x (num_layers - 1) -> linear -> output
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super(FFNModule, self).__init__()

        layers = []
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            # changes input size to hidden size after first layer
            input_size = hidden_size
        layers.append(torch.nn.Linear(hidden_size, output_size))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class FFNParams(BaseModel):
    input_size: int
    output_size: int
    hidden_size: int
    num_layers: int
    dropout: float

class TrainingParams(BaseModel):
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float

class FeatureNormPredictor(lightning.LightningModule):
    def __init__(self, ffn_params : FFNParams, training_params : TrainingParams):
        super().__init__()
        self.save_hyperparameters()
        self.ffn_params = ffn_params
        self.training_params = training_params
        self.model = FFNModule(**ffn_params.model_dump())
        self.loss_function = torch.nn.MSELoss()
        self.training_params = training_params

    def training_step(self, batch, batch_idx):
        x,y = batch
        outputs = self.model(x)
        loss = self.loss_function(outputs, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        outputs = self.model(x)
        loss = self.loss_function(outputs, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.model(batch)
    
    def predict(self, batch):
        return self.model(batch)
    
    def __call__(self, input):
        return self.model(input)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.training_params.learning_rate,
            weight_decay=self.training_params.weight_decay,
        )
        return optimizer
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))

    
class HiddenStateFeatureNormDataset(Dataset):
    def __init__(
        self, 
        input_embeddings: Dict[str, torch.Tensor],
        feature_norms: Dict[str, torch.Tensor],
    ):
        
        # Invariant: input_embeddings and target_feature_norms have exactly the same keys
        # this should be done by the train/test split and upstream data processing
        assert(input_embeddings.keys() == feature_norms.keys())

        self.words = list(input_embeddings.keys())
        self.input_embeddings = torch.stack([
            input_embeddings[word] for word in self.words
        ])
        self.feature_norms = torch.stack([
            feature_norms[word] for word in self.words
        ])
        
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        return self.input_embeddings[idx], self.feature_norms[idx]

# this is used when not optimizing
def train(args : Dict[str, Any]):

    # input_embeddings = torch.load(args.input_embeddings)
    # feature_norms = torch.load(args.feature_norms)
    # words = list(input_embeddings.keys())

    input_embeddings, feature_norms, norm_list = get_dict_pair(
        args.norm,
        args.norm_file,
        args.embedding_dir,
        args.lm_layer,
        translated= False if args.raw_buchanan else True,
        normalized= True if args.normal_buchanan else False
    ) 
    norms_file = open(args.save_dir+"/"+args.save_model_name+'.txt','w')
    norms_file.write("\n".join(norm_list))
    norms_file.close()

    words = list(input_embeddings.keys())
    
    model = FeatureNormPredictor(
        FFNParams(
            input_size=input_embeddings[words[0]].shape[0],
            output_size=feature_norms[words[0]].shape[0],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ),
        TrainingParams(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
    )

    # train/val split
    train_size = int(len(words) * 0.8)
    valid_size = len(words) - train_size
    train_words, validation_words = torch.utils.data.random_split(words, [train_size, valid_size])

    # TODO: Methodology Decision: should we be normalizing the hidden states/feature norms?
    train_embeddings = {word: input_embeddings[word] for word in train_words}
    train_feature_norms = {word: feature_norms[word] for word in train_words}
    validation_embeddings = {word: input_embeddings[word] for word in validation_words}
    validation_feature_norms = {word: feature_norms[word] for word in validation_words}

    train_dataset = HiddenStateFeatureNormDataset(train_embeddings, train_feature_norms)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_dataset = HiddenStateFeatureNormDataset(validation_embeddings, validation_feature_norms)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    callbacks = [
        lightning.pytorch.callbacks.ModelCheckpoint(
            save_last=True,
            dirpath=args.save_dir,
            filename=args.save_model_name,
        ),
    ]
    if args.early_stopping is not None:
        callbacks.append(lightning.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping,
            mode='min',
            min_delta=0.0
        ))

    #TODO Design Decision - other trainer args? Is device necessary?
    # cpu is fine for the scale of this model - only a few layers and a few hundred words
    trainer = lightning.Trainer(
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        accelerator="cpu",
        log_every_n_steps=7
    )

    trainer.fit(model, train_dataloader, validation_dataloader)

    val_metrics = trainer.validate(model, validation_dataloader)

    return model, val_metrics

# this is used when optimizing
def objective(trial: optuna.trial.Trial, args: Dict[str, Any]) -> float:
    # optimizing hidden size, batch size, and learning rate
    input_embeddings, feature_norms, norm_list = get_dict_pair(
        args.norm,
        args.norm_file,
        args.embedding_dir,
        args.lm_layer,
        translated= False if args.raw_buchanan else True,
        normalized= True if args.normal_buchanan else False
    )
    norms_file = open(args.save_dir+"/"+args.save_model_name+'.txt','w')
    norms_file.write("\n".join(norm_list))
    norms_file.close()

    words = list(input_embeddings.keys())
    input_size=input_embeddings[words[0]].shape[0]
    output_size=feature_norms[words[0]].shape[0]
    min_size = min(output_size, input_size)
    max_size = min(output_size, 2*input_size)if min_size == input_size else min(2*output_size, input_size)
    hidden_size = trial.suggest_int("hidden_size", min_size, max_size, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1, log=True)

    model = FeatureNormPredictor(
        FFNParams(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ),
        TrainingParams(
            num_epochs=args.num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
        ),
    )

    # train/val split
    train_size = int(len(words) * 0.8)
    valid_size = len(words) - train_size
    train_words, validation_words = torch.utils.data.random_split(words, [train_size, valid_size])

    train_embeddings = {word: input_embeddings[word] for word in train_words}
    train_feature_norms = {word: feature_norms[word] for word in train_words}
    validation_embeddings = {word: input_embeddings[word] for word in validation_words}
    validation_feature_norms = {word: feature_norms[word] for word in validation_words}

    train_dataset = HiddenStateFeatureNormDataset(train_embeddings, train_feature_norms)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_dataset = HiddenStateFeatureNormDataset(validation_embeddings, validation_feature_norms)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    callbacks = [
        # all trial models will be saved in temporary directory
        lightning.pytorch.callbacks.ModelCheckpoint(
            save_last=True,
            dirpath=os.path.join(args.save_dir,'optuna_trials'),
            filename="{}".format(trial.number)
        ),
        PyTorchLightningPruningCallback(
            trial,
            monitor='val_loss'
        )
    ]
    if args.early_stopping is not None:
        callbacks.append(lightning.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping,
            mode='min',
            min_delta=0.0
        ))
    # note that if optimizing is chosen, will automatically not implement vanilla early stopping 
    #TODO Design Decision - other trainer args? Is device necessary?
    
    if args.gpu is not None: # use gpu
        trainer = lightning.Trainer(
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=[args.gpu],
        log_every_n_steps=7,
        # enable_checkpointing=False
    ) 
    else: # use cpu
        trainer = lightning.Trainer(
            max_epochs=args.num_epochs,
            callbacks=callbacks,
            accelerator="cpu",
            log_every_n_steps=7,
            # enable_checkpointing=False
        )

    trainer.fit(model, train_dataloader, validation_dataloader)

    trainer.validate(model, validation_dataloader)
    
    return trainer.callback_metrics['val_loss'].item()

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    #TODO: Design Decision: Should we input paths, to the pre-extracted layers, or the model/layer we want to generate them from
    # required inputs
    parser.add_argument("--norm", type=str, required=True, help="feature norm set to use")
    parser.add_argument("--norm_file", type=str, required=True, help="csv containing norms")
    parser.add_argument("--embedding_dir", type=str, required=True, help="directory containing embeddings")
    parser.add_argument("--lm_layer", type=int, required=True, help="layer of embeddings to use")
    # optional accelerator
    parser.add_argument("--gpu",type=int, default=None, help="if using gpu, which device (used in optimizing only)")
    # if user selects optimize, hidden_size, batch_size and learning_rate will be optimized. 
    parser.add_argument("--optimize", action="store_true", help="optimize hyperparameters for training")
    parser.add_argument("--prune", action="store_true", help="prune unpromising trials when optimizing")
    # optional hyperparameter specs
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers in FFN")
    parser.add_argument("--hidden_size", type=int, default=100, help="hidden size of FFN")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate of FFN")
    # set this to at least 100 if doing early stopping
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay for training")
    parser.add_argument("--early_stopping", type=int, default=None, help="number of epochs to wait for early stopping")
    # optional dataset specs, for buchanan really
    parser.add_argument('--raw_buchanan', action="store_true", help="do not use translated values for buchanan")
    parser.add_argument('--normal_buchanan', action="store_true", help="use normalized features for buchanan")
    # required for output
    parser.add_argument("--save_dir", type=str, required=True, help="directory to save model to")
    parser.add_argument("--save_model_name", type=str, required=True, help="name of model to save")

    args = parser.parse_args()

    if args.early_stopping is not None:
        args.num_epochs = max(50, args.num_epochs)

    torch.manual_seed(10)

    if args.optimize:
        # call optimizer code here
        print("optimizing for learning rate, batch size, and hidden size")
        pruner = optuna.pruners.MedianPruner() if args.prune else optuna.pruners.NopPruner()
        sampler = optuna.samplers.TPESampler(seed=10)

        study = optuna.create_study(direction='minimize', pruner=pruner, sampler=sampler)
        study.optimize(partial(objective, args=args), n_trials = 100)

        print("Number of finished trials: {}".format(len(study.trials)))

        trial = study.best_trial
        print("Best trial: "+str(trial.number))
        
        print('saving best trial')
        for filename in os.listdir(os.path.join(args.save_dir,'optuna_trials')):
            if filename == "{}.ckpt".format(trial.number):
                shutil.move(os.path.join(args.save_dir,'optuna_trials',filename), os.path.join(args.save_dir, "{}.ckpt".format(args.save_model_name)))
        shutil.rmtree(os.path.join(args.save_dir,'optuna_trials'))

        # save validation loss in txt file
        val_loss = trial.value
    else:   
        model, val_metrics = train(args)
        val_loss = val_metrics[0]['val_loss']

    info_file = os.path.join(args.save_dir, 'train_info.csv')
    make_header = False
    if not os.path.exists(info_file) :
        make_header = True
    vals = open(info_file, "a")
    if make_header:
        vals.write("Name,Directory,Norm,EmbeddingDir,Layer,Optimize,Prune,NumLayers,HiddenSize,Dropout,NumEpochs,BatchSize,LearningRate,WeightDecay,EarlyStopping,RawBuchanan,NormalBuchanan,ValidationLoss\n")
    if args.optimize:
        vals.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(args.save_model_name, args.save_dir, args.norm, args.embedding_dir, args.lm_layer, 
                                                                                          args.optimize, args.prune, args.num_layers, trial.params['hidden_size'], args.dropout, args.num_epochs, trial.params['batch_size'], 
                                                                                          trial.params['learning_rate'], args.weight_decay, args.early_stopping, args.raw_buchanan, args.normal_buchanan, val_loss))
    else:
        vals.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(args.save_model_name, args.save_dir, args.norm, args.embedding_dir, args.lm_layer, 
                                                                                          args.optimize, args.prune, args.num_layers, args.hidden_size, args.dropout, args.num_epochs, args.batch_size, 
                                                                                          args.learning_rate, args.weight_decay, args.early_stopping, args.raw_buchanan, args.normal_buchanan, val_loss))
        