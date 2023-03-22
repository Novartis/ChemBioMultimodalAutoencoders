import torch
from multimodal_autoencoders.model.classifiers import SimpleClassifier, Discriminator

discriminator = Discriminator('adam', 0.001, 10, 2, 50)


class TestDiscriminator:
    
    def test_forward(self):
        latent = torch.rand(10, 10)
        scores = discriminator(latent)
        
        assert scores.size(dim=0) == 10
        assert scores.size(dim=1) == 2
        

classifier = SimpleClassifier('adam', 0.001, 10, 4)


class TestSimpleClassifier:
    
    def test_forward(self):
        latent = torch.rand(10, 10)
        scores = classifier.forward(latent)
        
        assert scores.size(dim=0) == 10
        assert scores.size(dim=1) == 4
