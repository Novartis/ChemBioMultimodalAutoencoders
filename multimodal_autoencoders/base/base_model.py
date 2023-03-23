from abc import abstractmethod
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


class ModelBase(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super().__init__()

        self.model = self._set_model()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    @abstractmethod
    def _set_model(self) -> nn.Module:
        """
        Abstract function; needs to be implemented in a child class.
        Returns the actual PyTorch module of the encoder e.g. from nn.Sequential().
        """
        raise NotImplementedError


class OptimizerBase:
    def __init__(
        self, parameters: Iterable[torch.nn.parameter.Parameter], optimizer: str, learning_rate: float, **kwargs
    ):
        print('Initializing optimizer')
        self._optimizer = self._set_optimizer(parameters, optimizer, learning_rate, **kwargs)

    def _set_optimizer(
        self, parameters: Iterable[torch.nn.parameter.Parameter], optimizer: str, learning_rate: float, **kwargs
    ):
        """
        Function to set the optimizer by string selection.

        optimizer: string for selecting optimizer [adam, sgd]
        learning_rate: optimizer learning rate
        kwargs: optimizer specific keywords

        return: optimizer
        """

        if optimizer == 'adam':
            return torch.optim.Adam(parameters, lr=learning_rate, **kwargs)
        elif optimizer == 'sgd':
            return torch.optim.SGD(parameters, lr=learning_rate, **kwargs)
        else:
            raise NotImplementedError('Provided optimizer name is not implemented. Select from: adam, sgd')

    def step(self):
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()


# base encoder class
class Encoder(ModelBase):
    def __init__(self, n_input: int, n_hidden: int):
        self.n_hidden: int = n_hidden
        self.n_input: int = n_input

        super().__init__()


# base decoder class
class Decoder(ModelBase):
    def __init__(self, n_input: int, n_hidden: int, n_z: int):
        self.n_input: int = n_input
        self.n_hidden: int = n_hidden
        self.n_z: int = n_z

        super().__init__()


# base classifier class
class Classifier(ModelBase, OptimizerBase):
    def __init__(self, optimizer: str, learning_rate: float, n_z: int, n_out: int, **kwargs):
        self.n_z = n_z
        self.n_out = n_out

        print('Initializing classifier model')
        # init model first
        ModelBase.__init__(self)

        # init optimizer second
        OptimizerBase.__init__(self, self.parameters(), optimizer, learning_rate, **kwargs)
