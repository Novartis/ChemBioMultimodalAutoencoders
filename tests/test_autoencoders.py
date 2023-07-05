import torch

from multimodal_autoencoders.base.autoencoder import VariationalAutoencoder
from multimodal_autoencoders.model.decoders import CpdDecoder, DynamicDecoder, PQSARDecoder
from multimodal_autoencoders.model.encoders import CpdEncoder, DynamicEncoder, PQSAREncoder

encoder = CpdEncoder(100, 50)
decoder = CpdDecoder(100, 50, 10)
vae = VariationalAutoencoder(encoder, decoder, 'adam', 0.001)


class TestCpdAutoencoder:
    def test_encode(self):
        """
        test encode function
        """

        batch = torch.ones(10, 100)

        z = vae.encode(batch)

        assert z.size(dim=1) == 10
        assert z.size(dim=0) == 10

    def test_decode(self):
        """
        test decode function
        """
        z = torch.rand(10, 10)
        decoded = decoder(z)

        assert decoded.size(dim=0) == 10
        assert decoded.size(dim=1) == 100

    def test_forward(self):
        """ """

        batch = torch.rand(10, 100)
        recon, latent, mu, logvar = vae(batch)

        assert latent.size(dim=0) == 10
        assert latent.size(dim=1) == 10


encoder = PQSAREncoder(100, 50)
decoder = PQSARDecoder(100, 50, 10)
vae = VariationalAutoencoder(encoder, decoder, 'adam', 0.001)


class TestPQSARAutoencoder:
    def test_encode(self):
        """
        test encode function
        """

        batch = torch.ones(10, 100)

        z = vae.encode(batch)

        assert z.size(dim=1) == 10
        assert z.size(dim=0) == 10

    def test_decode(self):
        """
        test decode function
        """
        z = torch.rand(10, 10)
        decoded = decoder(z)

        assert decoded.size(dim=0) == 10
        assert decoded.size(dim=1) == 100

    def test_forward(self):
        """ """

        batch = torch.rand(10, 100)

        recon, latent, mu, logvar = vae(batch)

        assert latent.size(dim=0) == 10
        assert latent.size(dim=1) == 10


encoder = DynamicEncoder(100, 50, 2)
decoder = DynamicDecoder(n_input=100, n_hidden=50, n_z=10, num_layers=2)
vae = VariationalAutoencoder(encoder, decoder, 'adam', 0.001)


class TestDynamicAutoencoder:
    def test_encode(self):
        """
        test encode function
        """

        batch = torch.ones(10, 100)

        z = vae.encode(batch)

        assert z.size(dim=1) == 10
        assert z.size(dim=0) == 10

    def test_decode(self):
        """
        test decode function
        """
        z = torch.rand(10, 10)
        decoded = decoder(z)

        assert decoded.size(dim=0) == 10
        assert decoded.size(dim=1) == 100

    def test_forward(self):
        """ """

        batch = torch.rand(10, 100)

        recon, latent, mu, logvar = vae(batch)

        assert latent.size(dim=0) == 10
        assert latent.size(dim=1) == 10
