import pytest
import numpy as np
import torch
import os
import shutil
from multimodal_autoencoders.joint_trainer import JointTrainer
from multimodal_autoencoders.base.autoencoder import VariationalAutoencoder
from multimodal_autoencoders.model.encoders import DynamicEncoder
from multimodal_autoencoders.model.decoders import DynamicDecoder
from multimodal_autoencoders.model.classifiers import SimpleClassifier, Discriminator


torch.autograd.set_detect_anomaly(True)

data_dict = {
    'mod1' : np.random.rand(10, 100),
    'mod2' : np.random.rand(10, 150),
    'mod3' : np.random.rand(10, 200),
}

cluster_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

a_model_dict = {
    'mod1': VariationalAutoencoder(
        DynamicEncoder(100, 50, 2), DynamicDecoder(100, 50, 10, 2), 'adam', 0.001, pretrain_epochs=2
    ),
    'mod2': VariationalAutoencoder(DynamicEncoder(150, 50, 4), DynamicDecoder(150, 50, 10, 4), 'adam', 0.001),
    'mod3': VariationalAutoencoder(DynamicEncoder(200, 50, 1), DynamicDecoder(200, 50, 10, 1), 'adam', 0.001)
}

b_model_dict = {
    'mod1': VariationalAutoencoder(
        DynamicEncoder(100, 50, 2), DynamicDecoder(100, 50, 10, 2), 'adam', 0.001, pretrain_epochs=2
    ),
    'mod2': VariationalAutoencoder(DynamicEncoder(150, 50, 4), DynamicDecoder(150, 50, 10, 4), 'adam', 0.001),
    'mod3': VariationalAutoencoder(DynamicEncoder(200, 50, 1), DynamicDecoder(200, 50, 10, 1), 'adam', 0.001)
}

disc_label_dict = {
    'mod1': torch.zeros(1,).long(),
    'mod2': torch.ones(1,).long(),
    'mod3': (torch.ones(1,).long() * 2)
}

disc_scores = torch.tensor([[0.33, 0.33, 0.33]])


@pytest.fixture
def trainer_no_class():
    trainer = JointTrainer(
        model_dict=a_model_dict,
        discriminator=Discriminator('adam', 0.001, 10, 3, 50),
    )
    return trainer


@pytest.fixture
def trainer_class():
    trainer = JointTrainer(
        model_dict=b_model_dict,
        discriminator=Discriminator('adam', 0.001, 10, 3, 50),
        classifier=SimpleClassifier('adam', 0.001, 10, 2)
    )
    return trainer


def test_compute_discriminator_loss(trainer_no_class):
    assert trainer_no_class._compute_discriminator_loss(disc_label_dict, disc_scores, 'mod1') > 0
    

def test_train_no_class(trainer_no_class):
    meter = trainer_no_class.train(
        train_data_dict=data_dict,
        val_data_dict=data_dict,
        max_epochs=5,
        batch_size=2,
        cluster_labels=cluster_labels,
        patience=4,
        min_value=100
    )
    # test that train is running at all

    assert len(meter) > 0

    key_checker = False
    for key in meter.keys():
        if 'val' in key:
            key_checker = True

    assert key_checker
    

def test_forward_no_class(trainer_no_class):
    recon_ar, latents_ar = trainer_no_class.forward('mod1', data_dict['mod1'], 1, False)
    
    assert recon_ar.shape[0] == data_dict['mod1'].shape[0]
    assert recon_ar.shape[1] == data_dict['mod1'].shape[1]
    
    assert latents_ar.shape[0] == 10
    assert latents_ar.shape[1] == 10
    

def test_translate_no_class(trainer_no_class):
    trans_ar = trainer_no_class.translate('mod1', 'mod2', data_dict['mod1'], 1, False)
    
    assert trans_ar.shape[0] == data_dict['mod2'].shape[0]
    assert trans_ar.shape[1] == data_dict['mod2'].shape[1]


def test_train_class(trainer_class):
    meter = trainer_class.train(
        train_data_dict=data_dict,
        val_data_dict=data_dict,
        max_epochs=5,
        batch_size=2,
        cl_weight=1,
        cluster_labels=cluster_labels,
        cluster_modality='mod1',
        patience=4,
        min_value=100
    )
    # test that train is running at all
    
    assert len(meter) > 0

    val_key_checker = False
    for key in meter.keys():
        if 'val' in key:
            val_key_checker = True

    assert val_key_checker

    clf_pretrain_key_checker = False
    for key in meter.keys():
        if 'pretrain_classifier_loss' in key:
            clf_pretrain_key_checker = True

    assert clf_pretrain_key_checker
    
    
def test_forward_class(trainer_class):
    recon_ar, latents_ar = trainer_class.forward('mod1', data_dict['mod1'], 1, False)
    
    assert recon_ar.shape[0] == data_dict['mod1'].shape[0]
    assert recon_ar.shape[1] == data_dict['mod1'].shape[1]
    
    assert latents_ar.shape[0] == 10
    assert latents_ar.shape[1] == 10


def test_translate_class(trainer_class):
    trans_ar = trainer_class.translate('mod1', 'mod2', data_dict['mod1'], 1, False)
    
    assert trans_ar.shape[0] == data_dict['mod2'].shape[0]
    assert trans_ar.shape[1] == data_dict['mod2'].shape[1]


def test_save_model(trainer_class):
    # save model
    trainer_class.save_model('./test_save')

    # check if file was created for each model
    assert os.path.exists('test_save/mod1/joint_vae.pth')
    assert os.path.exists('test_save/mod2/joint_vae.pth')
    assert os.path.exists('test_save/mod3/joint_vae.pth')

    # check if file was created for discriminator
    assert os.path.exists('test_save/discriminator/discriminator.pth')

    # check if file was created for classifier
    assert os.path.exists('test_save/classifier/classifier.pth')

    # check if file was created for loss weights
    assert os.path.exists('test_save/loss_weights.pth')

    # check if README file was creatd
    assert os.path.exists('test_save/README.txt')

    load_model_dict = {
        'mod1': VariationalAutoencoder(DynamicEncoder(100, 50, 2), DynamicDecoder(100, 50, 10, 2), 'adam', 0.001),
        'mod2': VariationalAutoencoder(DynamicEncoder(150, 50, 4), DynamicDecoder(150, 50, 10, 4), 'adam', 0.001),
        'mod3': VariationalAutoencoder(DynamicEncoder(200, 50, 1), DynamicDecoder(200, 50, 10, 1), 'adam', 0.001)
    }
    
    trainer = JointTrainer(
        model_dict=load_model_dict,
        discriminator=Discriminator('adam', 0.001, 10, 3, 50),
        classifier=SimpleClassifier('adam', 0.001, 10, 2),
        checkpoint_dir='./test_save'
    )

    # check if ae model can be loaded again
    assert len(trainer.forward('mod1', np.random.rand(10, 100))) == 2
    assert len(trainer.forward('mod2', np.random.rand(10, 150))) == 2
    assert len(trainer.forward('mod3', np.random.rand(10, 200))) == 2

    shutil.rmtree('./test_save')
