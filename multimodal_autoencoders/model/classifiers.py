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
    def __init__(
            self, optimizer: str, learning_rate: float,
            n_z: int, n_out: int, n_hidden: int, n_layer: int, **kwargs):
        
        self.n_hidden = n_hidden
        self.n_layer = n_layer

        super(Classifier, self).__init__(optimizer, learning_rate, n_z, n_out, **kwargs)
        
        def _set_model(self) -> nn.Module:

            module_list = nn.ModuleList()

            for i in range(self.num_layers):
                # check if we need to add the first layer
                # or a hidden layer
                if i == 0:
                    module_list.append(nn.Linear(self.n_input, self.n_hidden))
                else:
                    module_list.append(nn.Linear(self.n_hidden, self.n_hidden))
                
                module_list.append(nn.ReLU(inplace=True))

            return nn.Sequential(*module_list)
