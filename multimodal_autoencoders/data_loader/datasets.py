from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch

from multimodal_autoencoders.base.base_dataset import BaseDataset


class PairedDataset(BaseDataset):
    def __init__(self, data_dict: Dict[str, Iterable], cluster_labels: Union[np.array, None] = None):
        # parent class should take care of data needed for init
        super().__init__()

        # register data_dict internally
        self.data_dict = self._check_data_validity(data_dict)

        # store length as self value to be returned in required function
        self.length = self._set_length(self.data_dict)

        # set cluster labels object
        self.cluster_labels = self._set_cluster_labels(cluster_labels)

    def _set_cluster_labels(self, cluster_labels: Union[np.array, None] = None) -> np.ndarray:
        """ """
        # check if cluster info is available
        if cluster_labels is not None:
            # true: check that size is the same as data
            assert len(cluster_labels) == self.length
            labels = self.encode_cluster_labels(np.ravel(cluster_labels))
            return labels.reshape(self.length, 1)
        else:
            # false: create fake cluster info
            return np.zeros(self.length).reshape(self.length, 1)

    def _check_data_validity(self, data_dict: Dict[str, np.array]):
        """ """
        # check that all arrays have the same lengt
        len_list = [len(v) for v in data_dict.values()]
        assert len(set(len_list)) == 1, "Provided data sets don't have the same length."

        return data_dict

    def _set_length(self, data_dict: Dict[str, Any]) -> int:  # type: ignore
        """ """
        len_list = [len(v) for v in data_dict.values()]
        return len_list[0]

    def __getitem__(self, idx):
        """
        build return dict
        mapping key of data_dict to the appropriate tensor
        """
        sample_dict = {}
        for k, v in self.data_dict.items():
            sample_dict[k] = {
                'data': torch.from_numpy(v[idx]).float(),
                'cluster': torch.from_numpy(self.cluster_labels[idx]).int(),
            }

        return sample_dict


class UnpairedDataset(BaseDataset):
    def __init__(self, data_dict: Dict[str, np.array], cluster_labels: Optional[Dict[str, np.array]] = None):
        # parent class should take care of data needed for init
        super().__init__()

        # register data_dict internally
        self.data_dict = self._check_data_validity(data_dict)

        # store length as self value to be returned in required function
        self.length = self._set_length(self.data_dict)

        # set cluster labels object
        self.cluster_labels = self._set_cluster_labels(cluster_labels)

    def _set_cluster_labels(self, cluster_labels: Optional[Dict[str, np.array]] = None) -> Dict[str, np.array]:
        """ """
        cluster_dict = {}
        # check if cluster info is available
        if cluster_labels is not None:
            # true: check that we have cluster info for all data sets
            assert len(cluster_labels) == len(self.data_dict)

            k: str
            v: np.array
            for k, v in cluster_labels:  # type: ignore
                cluster_dict[k] = self.encode_cluster_labels(v)

            return cluster_dict

        else:
            # false: create fake cluster info
            for k, v in self.data_dict.items():
                cluster_dict[k] = np.zeros(len(v))

            return cluster_dict

    def _check_data_validity(self, data_dict: Dict[str, np.array]):
        """ """
        # do some check on the data

        return data_dict

    def _set_length(self, data_dict: Dict[str, np.array]):  # type: ignore
        """ """
        max_len = float('-inf')
        for v in data_dict.values():
            max_len = max(max_len, len(v))

        return max_len

    def __getitem__(self, idx):
        """
        build return dict
        mapping key of data_dict to the appropriate tensor
        """

        sample_dict = {}
        for k, v in self.data_dict.items():
            sample_dict[k] = {
                'data': torch.from_numpy(v[idx]).float(),
                'cluster': torch.from_numpy(self.cluster_labels[idx]).int(),
            }

        return sample_dict
