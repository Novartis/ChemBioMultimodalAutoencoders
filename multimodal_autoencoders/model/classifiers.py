import torch.nn as nn
from multimodal_autoencoders.base.base_model import Classifier


class Discriminator(Classifier):
    def __init__(
        self, optimizer: str, learning_rate: float,
        n_z: int, n_out: int, n_hidden: int, **kwargs):
        
        self.n_hidden = n_hidden
        
        super().__init__(optimizer, learning_rate, n_z, n_out, **kwargs)
        
        
    def _set_model(self):
        model = nn.Sequential(
            nn.Linear(self.n_z, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_out)
        )
        
        return model
    

class SimpleClassifier(Classifier):
    def __init__(
        self, optimizer: str, learning_rate: float, n_z: int, n_out: int):
        
        super().__init__(optimizer, learning_rate, n_z, n_out)

    def _set_model(self):
        
        model = nn.Linear(self.n_z, self.n_out)
        
        return model


class DynamicClassifier(Classifier):
    def __init__(self):
        super(Classifier, self).__init__()
        
        def _set_model(self) -> nn.Module:
            raise NotImplementedError