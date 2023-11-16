import torch
import lightning
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, TypedDict


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
    device: str


#TODO: rework this as pytorch-lightning module
class FeatureNormPredictor:
    """
    This class will load a dataset of hidden state representations of different words
    and their corresponding continuous linguistic feature norm vectors, then train a
    FFN to predict the continuous linguistic feature norm vector from the hidden state.
    """

    def __init__(
        self,
        input_embeddings: Dict[str, torch.Tensor],
        target_feature_norms: Dict[str, torch.Tensor],
        model_params: FFNParams,
        training_params: TrainingParams,
    ):
        # Invariant: input_embeddings and target_feature_norms have exactly the same keys
        # this should be done by the train/test split and upstream data processing
        assert(input_embeddings.keys() == target_feature_norms.keys())

        self.input_embeddings = input_embeddings
        self.target_feature_norms = target_feature_norms
        self.model = FFNModule(**model_params)
        self.training_params = training_params

    def train(self):
        vocabulary_order = list(self.input_embeddings.keys())
        inputs = torch.stack([
            self.input_embeddings[word] for word in vocabulary_order
        ])
        targets = torch.stack([
            self.target_feature_norms[word] for word in vocabulary_order
        ])
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.training_params["batch_size"],
            shuffle=True,
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_params["learning_rate"],
            weight_decay=self.training_params["weight_decay"],
        )
        # define the loss function with possible regularization
        loss_fn = torch.nn.MSELoss()

        for epoch in trange(self.training_params["num_epochs"]):
            for batch in dataloader:
                optimizer.zero_grad()
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.loss_fn = torch.nn.MSELoss()

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def predict(self, word: str):
        return self.model(self.input_embeddings[word])


#TODO: what should the interface look like?
class FeatureNormPredictor(lightning.LightningModule):
    def __init__(self, ffn_params : FFNParams):
        super().__init__()
        self.model = FFNModule(**FFNParams)
        self.loss_function = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x,y = batch
        outputs = self.model(x)
        loss = self.loss_function(outputs, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    
class HiddenStateFeatureNormDataset(Dataset):
    def __init__(
        self, 
        input_embeddings: Dict[str, torch.Tensor],
        feature_norms: Dict[str, torch.Tensor],
    ):
        
        # Invariant: input_embeddings and target_feature_norms have exactly the same keys
        # this should be done by the train/test split and upstream data processing
        assert(input_embeddings.keys() == feature_norms.keys())

        self.word_order = list(input_embeddings.keys())
        self.input_embeddings = torch.stack([
            input_embeddings[word] for word in self.word_order
        ])
        self.feature_norms = torch.stack([
            feature_norms[word] for word in self.word_order
        ])
        
    def __len__(self):
        return len(self.word_order)
    
    def __getitem__(self, idx):
        return self.input_embeddings[idx], self.feature_norms[idx]

        

    
