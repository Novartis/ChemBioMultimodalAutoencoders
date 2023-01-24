from cmath import tanh
import torch.nn as nn
from multimodal_autoencoders.base.base_model import Decoder

class CpdDecoder(Decoder):
    
    def _set_model(self) -> nn.Module:
        """
        Define actual decoder model
        """
        
        model = nn.Sequential(
            nn.Linear(self.n_z, self.n_input),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(self.n_input, self.n_input))
        
        return model


class PQSARDecoder(Decoder):
    
    def _set_model(self) -> nn.Module:
        """
        Define actual decoder model
        """
        
        model = nn.Sequential(
            nn.Linear(self.n_z, self.n_hidden),
            nn.LeakyReLU(0.1),
            #nn.Tanhshrink(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(0.1),
            #nn.Tanhshrink(),
            nn.Linear(self.n_hidden, self.n_input))
        
        return model

class HTSDecoder(Decoder):
    
    def _set_model(self) -> nn.Module:
        """
        Define actual decoder model
        """
        
        model = nn.Sequential(
            nn.Linear(self.n_z, self.n_hidden),
            nn.Tanhshrink(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Tanhshrink(),
            nn.Linear(self.n_hidden, self.n_input)
        )

        return model

class SimpleDecoder(Decoder):
    """Very basic decoder for small datasets

    Args:
        Decoder (_type_): _description_
    """
    
    def _set_model(self) -> nn.Module:
        """
        Define actual decoder model
        """
        
        model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.n_z, self.n_input)
        )

        return model

class LinearDecoder(Decoder):
    """Custom decoder to investigate usabilty of uncompressed
       HierVAE embeddings.

    Args:
        Decoder: Abstract base clase to implement
    """

    def _set_model(self) -> nn.Module:
        
        model = nn.Sequential(
            nn.Linear(self.n_input, self.n_input)
        )
        return model


class DynamicDecoder(Decoder):
    """_summary_

    Args:
        Decoder (_type_): _description_
    """

    def __init__(
        self, n_input: int, n_hidden: int, n_z: int, num_layers: int,
        dropout: float = 0.2, use_batchnorm: bool = True,
        activation: str = "lrelu"):
        """Decoder constructor

        Args:
            n_input (int): Number of dimenisons in the input
            n_hidden (int): Number of dimensions in the hidden layers
            n_z (int): Number of dimensions in the laten space
            num_layers (int): Number of layers to use
            dropout (float, optional): Percentage of dropout to use in each layer. Defaults to 0.2.
            use_batchnorm (bool, optional): If Batchnorm should be applied after each layer. Defaults to true.
            activation (str, optional): String of activation function to use. Chose from: "lrelu", "relu", "sigmoid", "tanhshrink". Defaults to "lrelu".
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

        super().__init__(n_input = n_input, n_hidden = n_hidden, n_z = n_z)

    
    def _set_model(self) -> nn.Module:

        module_list = nn.ModuleList()

        for i in range(self.num_layers - 1):
            # check if we need to add the first layer
            # or a hidden layer
            if i == 0:
                module_list.append(nn.Linear(self.n_z, self.n_hidden))
            else:
                module_list.append(nn.Linear(self.n_hidden, self.n_hidden))
            
            # add batchnorm if requested
            if self.use_batchnorm:
                module_list.append(nn.BatchNorm1d(self.n_hidden))
            
            # add dropout if requested
            if self.dropout > 0:
                module_list.append(nn.Dropout(p = self.dropout))

            module_list.append(self.activations[self.activation])

        # add output layer
        # be aware of edge case when the decoder only has one layer
        if self.num_layers == 1:
            module_list.append(nn.Linear(self.n_z, self.n_input))
        else:
            module_list.append(nn.Linear(self.n_hidden, self.n_input))

        return nn.Sequential(*module_list)
