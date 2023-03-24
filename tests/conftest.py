import pytest

from multimodal_autoencoders.base.autoencoder import VariationalAutoencoder
from multimodal_autoencoders.joint_trainer import JointTrainer
from multimodal_autoencoders.model.classifiers import Discriminator, SimpleClassifier
from multimodal_autoencoders.model.decoders import DynamicDecoder
from multimodal_autoencoders.model.encoders import DynamicEncoder

a_model_dict = {
    'mod1': VariationalAutoencoder(
        DynamicEncoder(100, 50, 2), DynamicDecoder(100, 50, 10, 2), 'adam', 0.001, pretrain_epochs=2
    ),
    'mod2': VariationalAutoencoder(DynamicEncoder(150, 50, 4), DynamicDecoder(150, 50, 10, 4), 'adam', 0.001),
    'mod3': VariationalAutoencoder(DynamicEncoder(200, 50, 1), DynamicDecoder(200, 50, 10, 1), 'adam', 0.001),
}

b_model_dict = {
    'mod1': VariationalAutoencoder(
        DynamicEncoder(100, 50, 2), DynamicDecoder(100, 50, 10, 2), 'adam', 0.001, pretrain_epochs=2
    ),
    'mod2': VariationalAutoencoder(DynamicEncoder(150, 50, 4), DynamicDecoder(150, 50, 10, 4), 'adam', 0.001),
    'mod3': VariationalAutoencoder(DynamicEncoder(200, 50, 1), DynamicDecoder(200, 50, 10, 1), 'adam', 0.001),
}


@pytest.fixture(scope='session', autouse=True)
def trainer_no_class():
    trainer = JointTrainer(
        model_dict=a_model_dict,
        discriminator=Discriminator('adam', 0.001, 10, 3, 50),
    )
    return trainer


@pytest.fixture(scope='session', autouse=True)
def trainer_class():
    trainer = JointTrainer(
        model_dict=b_model_dict,
        discriminator=Discriminator('adam', 0.001, 10, 3, 50),
        classifier=SimpleClassifier('adam', 0.001, 10, 2),
    )
    return trainer
