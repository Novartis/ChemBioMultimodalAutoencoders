from abc import abstractmethod
from typing import Any, Dict

import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

        # set up label encoder
        self.label_encoder = preprocessing.LabelEncoder()

        self.length: int = 0

    @abstractmethod
    def _set_cluster_labels(self, cluster_labels=None):
        """
        Abstract method: logic for setting the cluster labels
        variable and checking their validity
        needs to be implented by each subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _set_length(self, *input: Any) -> int:
        """
        Abstract method: logic for setting the length variable
        needs to be implented by each subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _check_data_validity(self, data_dict: Dict[str, np.array]):
        """
        Abstract method: logic for checking data validity
        needs to be implented by each subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """
        Return an appropriate training sample at idx.
        Needs to be implemnted by subclass.
        """
        raise NotImplementedError

    def encode_cluster_labels(self, cluster_labels: np.array):
        self.label_encoder.fit(cluster_labels)
        return self.label_encoder.transform(cluster_labels)

    def decode_cluster_labels(self, encoded_cluster_labels: np.array):
        return self.label_encoder.inverse_transform(encoded_cluster_labels)

    def __len__(self):
        return self.length
