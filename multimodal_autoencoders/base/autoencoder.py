import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from multimodal_autoencoders.base.base_model import Decoder, Encoder, OptimizerBase


# base autencoder class
class VariationalAutoencoder(nn.Module, OptimizerBase):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        optimizer: str,
        learning_rate: float,
        pretrain_epochs: int = 0,
        train_joint: bool = True,
        **kwargs,
    ):
        print('Initializing variational autoencoder model')
        # init torch.nn.Module first to make this a trainable module
        nn.Module.__init__(self)

        # register all parts of the model
        self._Encoder = self._validate_encoder(encoder)
        self._Decoder = self._validate_decoder(decoder)

        self.fc_mu = nn.Linear(encoder.n_hidden, decoder.n_z)
        self.fc_logvar = nn.Linear(encoder.n_hidden, decoder.n_z)

        # now we can init the associated optimizer, as the parameters are available
        OptimizerBase.__init__(self, self.parameters(), optimizer, learning_rate, **kwargs)

        # other class attributes
        self.pretrain_epochs: int = pretrain_epochs
        self.train_joint: bool = train_joint

    def get_encoder(self):
        return self._Encoder

    def get_decoder(self):
        return self._Decoder

    def encode(self, batch: torch.Tensor):
        hidden = self._Encoder(batch)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self._reparametrize(mu, logvar)
        return z

    def decode(self, z: torch.Tensor):
        return self._Decoder(z)

    def forward(self, batch: torch.Tensor):
        self.encode(batch)
        hidden = self._Encoder(batch)

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self._reparametrize(mu, logvar)

        res = self.decode(z)
        return res, z, mu, logvar

    def _reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        z = eps.mul(std).add_(mu)

        return z

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'

    def _validate_encoder(self, encoder: Encoder):
        """
        Ensure the passed object is a subclass of Encoder
        """
        assert isinstance(encoder, Encoder)
        return encoder

    def _validate_decoder(self, decoder: Decoder):
        """
        Ensure the passed object is a subclass of Decoder
        """
        assert isinstance(decoder, Decoder)
        return decoder
