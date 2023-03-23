import numpy as np
import pytest
import torch

from multimodal_autoencoders.data_loader.datasets import PairedDataset

data_dict = {
    'ar1': np.random.rand(10, 100),
    'ar2': np.random.rand(10, 100),
}
cluster_labels = np.random.rand(10)


@pytest.fixture
def pdsc():
    return PairedDataset(data_dict, cluster_labels)


@pytest.fixture
def pds():
    return PairedDataset(data_dict)


def test_length_pdsc(pdsc):
    assert len(pdsc) == 10


def test_getitem_pdsc(pdsc):
    loader = torch.utils.data.DataLoader(pdsc, batch_size=1, num_workers=0, shuffle=False)

    sample = next(iter(loader))

    # test that the dict has not more entries than anticipated
    assert len(sample) == 2

    # check if desired keys are available
    assert 'ar1' in sample.keys()
    assert 'ar2' in sample.keys()

    # are all items in the datadict respected
    assert 'data' in sample['ar1'].keys()
    assert 'cluster' in sample['ar1'].keys()

    assert 'data' in sample['ar2'].keys()
    assert 'cluster' in sample['ar2'].keys()

    # check that length of data and cluster annotation match
    assert len(sample['ar1']['data']) == len(sample['ar2']['data'])
    assert len(sample['ar1']['cluster']) == len(sample['ar2']['cluster'])

    assert len(sample['ar1']['data']) == len(sample['ar1']['cluster'])
    assert len(sample['ar2']['data']) == len(sample['ar2']['cluster'])
