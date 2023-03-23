##
# machine learning metric system adapted from TabNet:
# https://github.com/dreamquark-ai/tabnet/blob/bcae5f43b89fb2c53a0fe8be7c218a7b91afac96/pytorch_tabnet/metrics.py
##

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, roc_auc_score


@dataclass
class UnsupMetricContainer:
    """Container holding a list of metrics.

    Parameters
    ----------
    y_pred : torch.Tensor or np.array
        Reconstructed prediction (with embeddings)
    embedded_x : torch.Tensor
        Original input embedded by network
    obf_vars : torch.Tensor
        Binary mask for obfuscated variables.
        1 means the variables was obfuscated so reconstruction is based on this.

    """

    metric_names: List[str]
    prefix: str = ''

    def __post_init__(self):
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_pred, embedded_x, obf_vars):
        """Compute all metrics and store into a dict.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_pred : np.ndarray
            Score matrix or vector

        Returns
        -------
        dict
            Dict of metrics ({metric_name: metric_value}).

        """
        logs = {}
        for metric in self.metrics:
            res = metric(y_pred, embedded_x, obf_vars)
            logs[self.prefix + metric._name] = res
        return logs


@dataclass
class MetricContainer:
    """Container holding a list of metrics.

    Parameters
    ----------
    metric_names : list of str
        List of metric names.
    prefix : str
        Prefix of metric names.

    """

    metric_names: List[str]
    prefix: str = ''

    def __post_init__(self):
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_true, y_pred):
        """Compute all metrics and store into a dict.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_pred : np.ndarray
            Score matrix or vector

        Returns
        -------
        dict
            Dict of metrics ({metric_name: metric_value}).

        """
        logs = {}
        for metric in self.metrics:
            if isinstance(y_pred, list):
                res = np.mean([metric(y_true[:, i], y_pred[i]) for i in range(len(y_pred))])
            else:
                res = metric(y_true, y_pred)
            logs[self.prefix + metric._name] = res
        return logs


class Metric:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError('Custom Metrics must implement this function')

    @classmethod
    def get_metrics_by_names(cls, names):
        """Get list of metric classes.

        Parameters
        ----------
        cls : Metric
            Metric class.
        names : list
            List of metric names.

        Returns
        -------
        metrics : list
            List of metric classes.

        """
        available_metrics = cls.__subclasses__()
        available_names = [metric()._name for metric in available_metrics]
        metrics = []
        for name in names:
            assert name in available_names, f'{name} is not available, choose in {available_names}'
            idx = available_names.index(name)
            metric = available_metrics[idx]()
            metrics.append(metric)
        return metrics


class MSE(Metric):
    """
    Metric implementation of the torch mean squared error.
    """

    def __init__(self):
        self._name = 'mse'
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute MSE of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Prediction matrix or vector

        Returns
        -------
        float
            MSE of predictions vs targets.
        """

        mse_loss = nn.MSELoss()
        return mse_loss(y_true, y_score)


class CEL(Metric):
    """
    Cross Entropy Metric
    """

    def __init__(self):
        self._name = 'cel'
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute cross entropy loss of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            CE loss of predictions vs targets.
        """
        ce_loss = nn.CrossEntropyLoss()
        return ce_loss(y_true, y_score)


class AUC(Metric):
    """
    AUC.
    """

    def __init__(self):
        self._name = 'auc'
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute AUC of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            AUC of predictions vs targets.
        """
        return roc_auc_score(y_true, y_score[:, 1])


class Accuracy(Metric):
    """
    Accuracy.
    """

    def __init__(self):
        self._name = 'accuracy'
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute Accuracy of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            Accuracy of predictions vs targets.
        """
        y_pred = np.argmax(y_score, axis=1)
        return accuracy_score(y_true, y_pred)


class BalancedAccuracy(Metric):
    """
    Balanced Accuracy.
    """

    def __init__(self):
        self._name = 'balanced_accuracy'
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute Accuracy of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            Accuracy of predictions vs targets.
        """
        y_pred = np.argmax(y_score, axis=1)
        return balanced_accuracy_score(y_true, y_pred)


class MAE(Metric):
    """
    Mean Absolute Error.
    """

    def __init__(self):
        self._name = 'mae'
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute MAE (Mean Absolute Error) of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            MAE of predictions vs targets.
        """
        mae_loss = nn.L1Loss()
        return mae_loss(y_true, y_score)


class RMSE(Metric):
    """
    Root Mean Squared Error.
    """

    def __init__(self):
        self._name = 'rmse'
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute RMSE (Root Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            RMSE of predictions vs targets.
        """
        return np.sqrt(mean_squared_error(y_true, y_score))


class ACS(Metric):
    """
    Absolute Cosine
    """

    def __init__(self):
        self._name = 'cosine'
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute (Absolute Cosine Similarity) of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            MAE of predictions vs targets.
        """
        cos_loss = nn.CosineSimilarity()

        return torch.abs(cos_loss(y_true, y_score))


def check_metrics(metrics):
    """Check if custom metrics are provided.

    Parameters
    ----------
    metrics : list of str or classes
        List with built-in metrics (str) or custom metrics (classes).

    Returns
    -------
    val_metrics : list of str
        List of metric names.

    """
    val_metrics = []
    for metric in metrics:
        if isinstance(metric, str):
            val_metrics.append(metric)
        elif issubclass(metric, Metric):
            val_metrics.append(metric()._name)
        else:
            raise TypeError('You need to provide a valid metric format')
    return val_metrics
