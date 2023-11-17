import torch
import lightning
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, TypedDict
import argparse


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
    
class FFNParams(TypedDict):
    input_size: int
    output_size: int
    hidden_size: int
    num_layers: int
    dropout: float

class TrainingParams(TypedDict):
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float

class FeatureNormPredictor(lightning.LightningModule):
    def __init__(self, ffn_params : FFNParams, training_params : TrainingParams):
        super().__init__()
        self.model = FFNModule(**FFNParams)
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
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.model(batch)
    
    def predict(self, batch):
        return self.model(batch)
    
    def __call__(self, input):
        return self.model(input)
    
    def configure_optimizer(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.training_params["learning_rate"],
            weight_decay=self.training_params["weight_decay"],
        )
    
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

def train(args : Dict[str, Any]):

    input_embeddings = torch.load(args.input_embeddings)
    feature_norms = torch.load(args.feature_norms)
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

    train_size = int(len(words) * 0.8)
    valid_size = len(words) - train_size
    train_words, validation_words = torch.utils.random_split(words, [train_size, valid_size])

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

    #TODO Design Decision - other trainer args? Is device necessary?
    trainer = lightning.Trainer(
        max_epochs=args.num_epochs,
    )

    trainer.fit(model, train_dataloader)

    trainer.validate(model, validation_dataloader)

    model.save_model(args.save_path)

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    #TODO: Design Decision: Should we input paths, to the pre-extracted layers, or the model/layer we want to generate them from
    parser.add_argument("--input_embeddings", type=str, required=True, help="path to input hidden states")
    parser.add_argument("--feature_norms", type=str, required=True, help="path to feature norms")
    parser.add_argument("--layers", type=int, default=2, help="number of layers in FFN")
    parser.add_argument("--hidden_size", type=int, default=100, help="hidden size of FFN")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate of FFN")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay for training")
    parser.add_argument("--save_path", type=str, required=True, help="path to save model to")

    args = parser.parse_args()
    train(args)