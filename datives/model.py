import torch
import lightning
from pydantic import BaseModel

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
        