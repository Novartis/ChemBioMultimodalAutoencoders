import torch.nn as nn
from multimodal_autoencoders.base.base_model import Encoder


class CpdEncoder(Encoder):
    
    def _set_model(self) -> nn.Module:
        """
        Return actual encoder architecture that should be used.
        """
        
        # Encoder
        model = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(self.n_hidden),
            nn.Dropout(0.3),
            nn.Linear(self.n_hidden, self.n_hidden),
        )
        
        return model
    
    
class PQSAREncoder(Encoder):
    
    def _set_model(self) -> nn.Module:
        """
        """
        
        model = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(self.n_hidden),
            nn.Dropout(0.3),
            nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden)
        )
        
        return model


class HTSEncoder(Encoder):

    def _set_model(self) -> nn.Module:
        """_summary_

        Returns:
            nn.Module: _description_
        """

        model = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.Tanhshrink(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Tanhshrink(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Tanhshrink()
        )
        
        return model


class SimpleEncoder(Encoder):
    """Very simple encoder for small datasets

    Args:
        Encoder (_type_): _description_
    """
    
    def _set_model(self) -> nn.Module:
        """
        """
        
        model = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        return model


class LinearEncoder(Encoder):
    """Custom class to investigate useability of uncompressed HierVAE embeddings.

    Args:
        Encoder: Abstract parent class to implement
    """

    def _set_model(self) -> nn.Module:
        
        model = nn.Sequential(
            nn.Linear(self.n_input, self.n_input)
        )
        return model


class DynamicEncoder(Encoder):
    """Custom ecnoder class that is fully parameterizable.

    args:
        encoder (_type_): Subclass of base encoder
    """

    def __init__(
        self, n_input: int, n_hidden: int, num_layers: int,
        dropout: float = 0.2, use_batchnorm: bool = True,
        activation: str = 'lrelu'
    ):
        """Encoder constructor

        Args:
            n_input (int): Number of dimenisons in the input
            n_hidden (int): Number of dimensions in the hidden layers
            num_layers (int): Number of layers to use
            dropout (float, optional): Percentage of dropout to use in each layer. Defaults to 0.2.
            use_batchnorm (bool, optional): If Batchnorm should be applied after each layer. Defaults to true.
            activation (str, optional):
                String of activation function to use.
                Chose from: "lrelu", "relu", "sigmoid", "tanhshrink". Defaults to "lrelu".
        """
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.activations = {
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanhshrink': nn.Tanhshrink(),
            'tanh': nn.Tanh()
        }

        super().__init__(n_input=n_input, n_hidden=n_hidden)

    def _set_model(self) -> nn.Module:
        print(self.num_layers)

        module_list = nn.ModuleList()

        for i in range(self.num_layers):
            # check if we need to add the first layer
            # or a hidden layer
            if i == 0:
                module_list.append(nn.Linear(self.n_input, self.n_hidden))
            else:
                module_list.append(nn.Linear(self.n_hidden, self.n_hidden))
            
            # add batchnorm if requested
            if self.use_batchnorm:
                module_list.append(nn.BatchNorm1d(self.n_hidden))
            
            # add dropout if requested
            if self.dropout > 0:
                module_list.append(nn.Dropout(p=self.dropout))

            module_list.append(self.activations[self.activation])

        return nn.Sequential(*module_list)
