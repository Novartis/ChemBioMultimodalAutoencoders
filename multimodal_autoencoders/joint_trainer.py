import copy
import heapq
import logging
import os
import pprint
import shutil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.autograd import Variable

from multimodal_autoencoders.base.autoencoder import VariationalAutoencoder
from multimodal_autoencoders.base.base_model import Classifier
from multimodal_autoencoders.data_loader.datasets import PairedDataset, UnpairedDataset
from multimodal_autoencoders.model.metric import *  # noqa: F403


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class JointTrainer:
    def __init__(
        self,
        model_dict: Dict[str, VariationalAutoencoder],
        discriminator: Classifier,
        classifier: Optional[Classifier] = None,
        checkpoint_dir: str = '',
    ):
        # set up logging. move maybe to util or main file later
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.info('initializing trainer')

        # init device flag
        self.device = 'cpu'

        # register autoencoders and discriminator
        self.model_dict = model_dict
        self.discriminator = discriminator

        self.use_classifier = False
        if classifier is not None:
            # check for instance of correct class
            assert isinstance(classifier, Classifier), 'Provided object is not of class Classifier'
            self.classifier = classifier
            self.use_classifier = True

        # check if model should be resumed from file
        if len(checkpoint_dir) > 0:
            self._resume_from_file(checkpoint_dir)

        # define dict of metrics to track
        self.meter_dict: Dict[str, AverageMeter] = {}

        # register loss functions
        self._ae_recon_metric = MSE()
        self._anchor_metric = MAE()
        self._classification_metric = CEL()

    def _resume_from_file(self, checkpoint_dir: str) -> None:
        logging.info('loading models from checkpoint')

        # resume autoencoders
        for key, model in self.model_dict.items():
            vae_dict_path = os.path.join(checkpoint_dir, key, 'joint_vae.pth')

            # make sure the path exists
            assert os.path.exists(vae_dict_path), f'Path for model {key} does not exist.'

            vae_state_dict = torch.load(vae_dict_path, map_location=torch.device('cpu'))
            model.load_state_dict(vae_state_dict['autoencoder'])
            model._optimizer.load_state_dict(vae_state_dict['optimizer'])

        # resume discriminator
        # make sure path exists
        discriminator_path = os.path.join(checkpoint_dir, 'discriminator', 'discriminator.pth')
        assert os.path.exists(discriminator_path), 'Discriminator object provided, but path does not exist.'
        self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=torch.device('cpu')))

        # resume classifier
        if self.classifier is not None:
            classifier_path = os.path.join(checkpoint_dir, 'classifier', 'classifier.pth')
            assert os.path.exists(classifier_path), 'Classifier object provided, but path does not exist.'
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=torch.device('cpu')))

        # resume loss weights
        loss_weight_path = os.path.join(checkpoint_dir, 'loss_weights.pth')
        assert os.path.exists(loss_weight_path)
        loss_weight_dict = torch.load(loss_weight_path)

        # overwirte weights with content from stored dict
        self.recon_weight_dict = loss_weight_dict['recon_weight']
        self.beta = loss_weight_dict['beta']
        self.cl_weight = loss_weight_dict['cl_weight']
        self.disc_weight = loss_weight_dict['disc_weight']
        self.anchor_weight = loss_weight_dict['anchor_weight']
        self.paired = loss_weight_dict['paired']

    def _set_recon_weight_dict(self, recon_weight: Union[float, Dict[str, float]]) -> Dict[str, float]:
        """Helper function to create a model reconstruction weight dictionary if a single weight value
            was given. If already dict, just use as is.

        Args:
            recon_weight (Union[float, Dict[str, float]]): _description_

        Returns:
            Dict[str, float]: _description_
        """

        recon_weight_dict = {}
        if isinstance(recon_weight, float):
            for key in self.model_dict.keys():
                recon_weight_dict[key] = recon_weight
        elif isinstance(recon_weight, dict):
            recon_weight_dict = recon_weight
        else:
            raise NotImplementedError

        return recon_weight_dict

    def train(
        self,
        train_data_dict: Dict[str, np.array],
        val_data_dict: Dict[str, np.array],
        max_epochs: int,
        batch_size: int = 128,
        recon_weight: Union[float, Dict[str, float]] = 1,
        beta: float = 0.01,
        disc_weight: float = 1,
        anchor_weight: float = 0.001,
        cl_weight: float = 0,
        ae_metric: Optional[Any] = None,
        anchor_metric: Optional[Any] = None,
        classifier_metric: Optional[Any] = None,
        cluster_labels: Optional[Union[Dict[str, np.array], np.array]] = None,
        cluster_modality: str = '',
        log_path: str = '',
        use_gpu: bool = False,
        patience: int = -1,
        min_value: float = 0,
    ):
        """_summary_

        Args:
            train_data_dict (Dict[str, np.array]):
                dictionary mapping modality names to numpy arrays containing training data
            val_data_dict (Dict[str, np.array]):
                dictionary mapping modality names to numpy arrays containing training data
            max_epochs (int): Maximal number of epochs the model should run.
            batch_size (int, optional): Number of samples per batch. Defaults to 128.
            recon_weight (Union[float, Dict[str, float]], optional):
                Scaling values for reconstruction loss. Defaults to 1.
            beta (float, optional): Sacling value for KL divergence. Defaults to 0.01.
            disc_weight (float, optional): Scaling value for the discriminator loss. Defaults to 1.
            anchor_weight (float, optional): Scaling value for the anchor loss. Defaults to 0.001.
            cl_weight (float, optional): Scaling value for the classifier loss. Defaults to 0.
            ae_metric (Metric, optional): Metric to use as the reconstruction metric. Defaults to MSE().
            anchor_metric (Metric, optional): Metric to use as the anchor metric. Defaults to MAE().
            classifier_metric (Metric, optional): Metric to use as the classifier metric. Defaults to CEL().
            cluster_labels (Union[Dict[str, np.array], np.array, None], optional):
                Cluster labels to use during training if provided. Defaults to None.
            cluster_modality (str, optional):
                Modality which should use the classifier during pre-training. Defaults to "".
            log_path (str, optional): Path to write training log to. If omitted log will spill to console.
            use_gpu (bool, optional): Should GPU acceleration be used during training. Defaults to False.
            patience (int, optional):
                Number of epochs to wait before training is stopped in early stoppping. No early stopping if omitted.
            min_value (float, optional):
                Minimal loss difference needed to be exceeded for early stopping.
                Only effective if patience is provided.

        Returns:
            meter dictionary: Dictionary of loss value meters for training and validation.
        """

        logging.info('setting loss scalings and other parameters')
        # internal flags and parameters
        self.max_epochs = max_epochs

        # loss weights
        self.recon_weight_dict = self._set_recon_weight_dict(recon_weight)
        self.beta = beta
        self.cl_weight = cl_weight
        self.disc_weight = disc_weight
        self.anchor_weight = anchor_weight

        # dynamically derived parameters
        self.paired = False
        if anchor_weight > 0:
            self.paired = True

        # check for custom metrics or use defaults
        if ae_metric is not None:
            self._ae_recon_metric = ae_metric
        if anchor_metric is not None:
            self._anchor_metric = anchor_metric
        if classifier_metric is not None:
            self._classification_metric = classifier_metric

        # check for cluster labels
        if cluster_labels is not None:
            self._assert_cluster_labels(self.classifier, cluster_labels)
            # check if cluster modality tag provided and matches with a model key
            if len(cluster_modality) > 0:
                assert (
                    cluster_modality in self.model_dict.keys()
                ), 'Provided cluster modality flag has no match in model dictionary.'

        # checking device status
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

        # move models to appropriate device
        self._dict_to_device(self.model_dict)
        self.discriminator = self.discriminator.to(self.device)

        # check classifier status
        if self.use_classifier:
            self._assert_cluster_labels(self.classifier, cluster_labels)
            # check if cluster modality tag provided and matches with a model key
            if len(cluster_modality) > 0:
                assert (
                    cluster_modality in self.model_dict.keys()
                ), 'Provided cluster modality flag has no match in model dictionary.'

            # move classifier to device
            self.classifier.to(self.device)

        # set up a data loader from the input data
        train_dataloader = self._set_up_dataloader(
            train_data_dict, batch_size, shuffle=True, cluster_labels=cluster_labels
        )

        # make sure the data and model keys match
        # intersect key sets
        key_intersection = self.model_dict.keys() & train_data_dict.keys()
        # assert that intersection has the same size as model dict
        assert len(key_intersection) == len(self.model_dict), 'Keys of data and model dictionary are not matching.'

        # check if early stopping vars provided correctly
        if min_value > 0:
            assert (
                min_value > 0 and patience > 0
            ), 'Min value provided without positive patience. Please provide a positive integer patience.'
        assert patience < self.max_epochs, 'Patience larger than maximal number of epochs.'

        # setting variables
        norm_factor = 1 / len(self.model_dict)
        min_val_breaches = 0
        ckp_heap: Any[str] = []
        heapq.heapify(ckp_heap)

        # logic for AE pretraining
        self._pretrain_ae(train_dataloader, cluster_modality, log_path)

        # tracking variable for early stopping
        prev_val_loss = 10

        logging.info('starting joint training')
        # main training loop
        for epoch in range(self.max_epochs):
            train_meter_dict: Dict[str, AverageMeter] = {}

            for _idx, batch in enumerate(train_dataloader):
                # train ae
                # set autoencoders into trainig mode
                self._set_train_models()

                # set discriminator eval mode
                self.discriminator.eval()

                # optional: set classifier to trainig
                if self.use_classifier:
                    self.classifier.train()

                # resetting gradients
                self._zero_grad_autoencoders()

                if self.use_classifier:
                    self.classifier.zero_grad()

                # extract ae forwad runs to additional function and store in dict
                forward_pass = self._ae_forward_pass(batch)

                # get fake discrimnator labels
                disc_label_dict = self._build_disc_label_dict(batch)

                # pass on to function to compute different loss parts
                model_loss_dict, classifier_loss_dict = self._compute_model_loss(
                    batch, forward_pass, disc_label_dict, train_meter_dict
                )

                # build weighted sum loss over all individual models
                loss = sum(norm_factor * value for value in model_loss_dict.values())

                # check that the loss is not na
                assert not torch.isnan(loss)

                # update epoch loss tracker
                self._update_meter_dict(train_meter_dict, 'loss', loss)

                # back propagation
                loss.backward()
                # step autoencoders
                self._step_autoencoders()

                # optional: step classifier
                if self.use_classifier:
                    self.classifier.step()

                    # compute total classifier loss for tracking only
                    # if we are using the classifier its loss is already
                    # accounted for in the model_loss
                    classifier_loss = sum(norm_factor * value for value in classifier_loss_dict.values())
                    self._update_meter_dict(train_meter_dict, 'classifier_loss', classifier_loss)

                # train discriminator

                # set ae models to eval
                self._set_eval_models()

                # optional: set classifier to eval
                if self.use_classifier:
                    self.classifier.eval()

                # pass on to function to train discriminator
                discriminator_loss_dict = self._train_discriminator_epoch(forward_pass, disc_label_dict)

                # build weighted sum of discriminator loss
                loss = sum(norm_factor * value for value in discriminator_loss_dict.values())
                self._update_meter_dict(train_meter_dict, 'discriminator_training_loss', loss)

                # backpropegate discriminator
                loss.backward()
                self.discriminator.step()

            # check validation performance
            val_meter_dict = self._val(val_data_dict, batch_size=batch_size)

            # track sum of recon loss
            val_sum_recon_loss = sum(
                norm_factor * val_meter_dict[f'{model_key}_recon'].avg for model_key in self.model_dict.keys()
            )

            # compute difference between previous and current val_recon loss
            loss_diff = prev_val_loss - val_sum_recon_loss

            # update difference meter
            self._update_meter('val_sum_recon_loss', val_sum_recon_loss)
            self._update_meter('val_recon_loss_diff', loss_diff)

            # update tracking variable
            # current value will be previous in next epoch
            prev_val_loss = val_sum_recon_loss

            # transfer train meters to global tracker
            for key, value in train_meter_dict.items():
                self._update_meter(key, value.avg)

            # transfer val meters to global tracker
            for key, value in val_meter_dict.items():
                self._update_meter(f'val_{key}', value.avg)

            # write current meters to meter log
            self._write_meter_log(epoch, log_path)

            # handle early stopping
            es_result = self._early_stopping(ckp_heap, patience, loss_diff, min_value, min_val_breaches, epoch)

            if isinstance(es_result, int):
                min_val_breaches = es_result
            else:
                # meter dict was returned from early stopping
                # patience was breached
                # break the training loop
                break

        return self.meter_dict

    def _assert_cluster_labels(self, classifier, cluster_labels):
        if classifier is not None:
            # check that we also have labels
            assert cluster_labels is not None, 'Classifier provided, but no cluster labels were found'
            # check that loss weight is > 0
            assert self.cl_weight > 0, 'Classifier provided, but loss weight still 0. Increase cl_weight parameter.'

    def _set_up_dataloader(
        self,
        data_dict: Dict[str, np.array],
        batch_size: int,
        shuffle: bool,
        cluster_labels: Union[Dict[str, np.array], np.array, None] = None,
    ) -> torch.utils.data.DataLoader:
        """ """
        # check that we really got a dictionary
        assert isinstance(data_dict, dict)

        # check if models should expect paired data
        dataset = None
        if self.paired:
            dataset = PairedDataset(data_dict, cluster_labels)
        else:
            dataset = UnpairedDataset(data_dict, cluster_labels)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=shuffle)

        return data_loader

    def _early_stopping(
        self,
        ckp_heap: List[Optional[str]],
        patience: int,
        loss_diff: float,
        min_value: float,
        min_val_breaches: int,
        epoch: int,
    ) -> Union[int, Dict[str, AverageMeter]]:
        """_summary_

        Args:
            ckp_heap (Iterable): _description_
            patience (int): _description_
            loss_diff (float): _description_
            min_value (float): _description_
            min_val_breaches (int): _description_

        Returns:
            _type_: _description_
        """
        # check if we want to use early stopping
        # and validation performance is worse than training performance
        # and required minimal difference was not reached anymore
        if patience > 0 and loss_diff < min_value:
            # track number of min value breaches
            min_val_breaches += 1

            # store checkpoint to return to if needed
            # spill into current directory
            # delete all that are not needed anymore
            curr_ckpt_dir = f'epoch_{epoch}_ckpt/'
            self.save_model(curr_ckpt_dir)

            # put checkpoint into heap
            if len(ckp_heap) > patience:
                # only keep as many checkpoints as we
                # have patience epochs

                # delete the checkpoint we don't need anymore
                earliest_ckp = heapq.heappop(ckp_heap)
                shutil.rmtree(f'./{earliest_ckp}')

            # add newest checkpoint to heap
            heapq.heappush(ckp_heap, curr_ckpt_dir)

            # check if patience was breached
            if min_val_breaches > patience:
                # write log message and return
                logging.info(f'Stopping training early. Patience of {patience} epochs was reached.')

                # delete all checkpoints that overfitted
                for i in range(1, len(ckp_heap)):
                    shutil.rmtree(f'./{ckp_heap[i]}')

                return self.meter_dict

            return min_val_breaches

        else:
            # reset breaches, as we only want to stop on conescutive overfitting
            return 0

    def _pretrain_ae(
        self, train_dataloader: torch.utils.data.DataLoader, cluster_modality: str = '', log_path: str = ''
    ) -> None:
        """
        Pre-training loop. Trains autoencoders for number of epochs stated in pretrain_epochs attribtute
        of each autoencoder instance.
        """

        for model_key, model in self.model_dict.items():
            if model.pretrain_epochs == 0:
                continue

            # set model to training mode
            model.train()

            # optional: set classifier to trainig
            if self.use_classifier and model_key == cluster_modality:
                self.classifier.train()

            # make a copy of the dataloader as to not modify it
            pretrain_dataloader = copy.deepcopy(train_dataloader)

            epoch_count = 0

            while model.pretrain_epochs > 0:
                logging.info(f'Pretrain epoch {model.pretrain_epochs}')

                # set up local meter dict
                pretrain_meter_dict: Dict[str, AverageMeter] = {}

                for _idx, batch in enumerate(pretrain_dataloader):
                    # check if model still has pretrain epochs left

                    X = self._batchtensor_to_device(batch[model_key]['data'])
                    # ae forward pass
                    recon, latents, mu, logvar = model(X)

                    # compute loss values
                    # ae recon loss
                    recon_loss = self._ae_recon_metric(X, recon)
                    self._update_meter_dict(pretrain_meter_dict, f'{model_key}_pretrain_recon', recon_loss)

                    # kl loss
                    kl_div = self._compute_KL_loss(mu, logvar)
                    self._update_meter_dict(pretrain_meter_dict, f'{model_key}_pretrain_kl', float(torch.mean(kl_div)))

                    # build weighted sum loss over all individual models
                    loss = self.recon_weight_dict[model_key] * recon_loss + self.beta * kl_div

                    # optional: classifier loss
                    if self.use_classifier and model_key == cluster_modality:
                        classifier_scores = self.classifier(latents)
                        # transform cluster information from multidimensional (one row per batch)
                        # to a single column vector
                        classifier_loss = self._classification_metric(
                            classifier_scores, self._batchtensor_to_device(batch[model_key]['cluster'].flatten().long())
                        )
                        self._update_meter_dict(
                            pretrain_meter_dict,
                            f'{model_key}_pretrain_classifier_loss',
                            classifier_loss.detach().clone(),
                        )

                        # add classifier loss to model pretraining loss
                        loss += self.cl_weight * classifier_loss

                    # check that the loss is not na
                    assert not torch.isnan(loss), 'NA found in pretraining loss.'

                    # log total loss
                    self._update_meter_dict(pretrain_meter_dict, f'{model_key}_pretrain_loss', loss)

                    # back propagation
                    loss.backward()

                    # step autoencoders and zero optimizers
                    model.step()

                    # reset gradient
                    model.zero_grad()

                    # optional: step classifier
                    if self.use_classifier and model_key == cluster_modality:
                        self.classifier.step()
                        self.classifier.zero_grad()

                model.pretrain_epochs = model.pretrain_epochs - 1

                # transfer averages to global meters
                for key, value in pretrain_meter_dict.items():
                    self._update_meter(key, value.avg)

                # write to log
                self._write_meter_log(epoch_count, f'{log_path}.{model_key}')
                # update counter
                epoch_count += 1

            # delete copy
            del pretrain_dataloader

    def _dict_to_device(self, dictionary: Dict):
        for key, _value in dictionary.items():
            dictionary[key] = dictionary[key].to(self.device)

    def _ae_forward_pass(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        """ """
        forward_result_dict = {}

        # loop through all registered models
        for key, model in self.model_dict.items():
            # ae forward pass
            recon, latents, mu, logvar = model(self._batchtensor_to_device(batch[key]['data']))

            # store forward pass results to dict
            forward_result_dict[key] = [recon, latents, mu, logvar]

        return forward_result_dict

    def _build_disc_label_dict(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """_summary_

        Args:
            batch (Dict[str, Dict[str, torch.Tensor]]): _description_

        Returns:
            Dict[str, torch.Tensor]: _description_
        """

        disc_label_dict = {}
        # set up fake labels for discriminator for a given batch
        for idx, key in enumerate(self.model_dict.keys()):
            curr_batch_size = len(batch[key]['data'])

            # create appropriate tensor and move to device
            label_tensor = (
                torch.ones(
                    curr_batch_size,
                ).long()
                * idx
            )
            disc_label_dict[key] = self._batchtensor_to_device(label_tensor)

        return disc_label_dict

    def _compute_model_loss(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
        forward_pass: Dict[str, List[torch.Tensor]],
        disc_label_dict: Dict[str, torch.Tensor],
        meter_dict: Dict[str, AverageMeter],
    ) -> tuple:
        """
        Main function to get autoencoder loss values.
        Function name could be improved.

        return: dict with ae model loss, dict with classifier model loss (empty if classifier not used)
        """

        model_loss_dict = {}
        classifier_loss_dict = {}

        # loop through per model forward pass results
        for key, forward_result in forward_pass.items():
            recon = forward_result[0]
            latents = forward_result[1]
            mu = forward_result[2]
            logvar = forward_result[3]

            # TODO: add check if discriminator should use cluster labels
            discrimnator_scores = self.discriminator(latents)

            # optional: get classifier scores
            if self.use_classifier:
                classifier_scores = self.classifier(latents)

            # compute loss values
            # ae recon loss
            recon_loss = self._ae_recon_metric(self._batchtensor_to_device(batch[key]['data']), recon)

            self._update_meter_dict(meter_dict, f'{key}_recon', recon_loss)

            # kl loss
            kl_div = self._compute_KL_loss(mu, logvar)
            self._update_meter_dict(meter_dict, f'{key}_kl', kl_div)

            # discriminator loss
            discriminator_loss = self._compute_discriminator_loss(disc_label_dict, discrimnator_scores, key)
            self._update_meter_dict(meter_dict, 'discriminator_adverserial_loss', discriminator_loss)

            # translation loss (for tracking)
            trans_loss = self._track_translation(key, forward_pass, batch)
            self._update_meter_dict(meter_dict, f'{key}_translation_loss', trans_loss)

            # optional: classifier loss
            if self.use_classifier:
                # transform cluster information from multidimensional (one row per patch) to a single column vector
                classifier_loss = self._classification_metric(
                    classifier_scores, self._batchtensor_to_device(batch[key]['cluster'].flatten().long())
                )
                classifier_loss_dict[key] = classifier_loss.detach().clone()

            # build per model sum
            model_loss = (
                self.recon_weight_dict[key] * recon_loss + self.disc_weight * discriminator_loss + self.beta * kl_div
            )

            if self.use_classifier:
                # add classifier loss to per model loss
                model_loss += self.cl_weight * classifier_loss

            # optional: anchor loss (check if we are doing paired?)
            if self.paired:
                anchor_loss = self._compute_anchor_loss(key, forward_pass)
                self._update_meter_dict(meter_dict, 'anchor_loss', anchor_loss)

                model_loss += self.anchor_weight * anchor_loss

            # store loss under model key
            model_loss_dict[key] = model_loss

        return model_loss_dict, classifier_loss_dict

    def _compute_KL_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        KLloss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLloss

    def _compute_discriminator_loss(
        self, disc_label_dict: Dict[str, torch.Tensor], discriminator_scores: torch.Tensor, key: str
    ) -> float:
        model_dis_loss = 0
        for other_key in disc_label_dict.keys():
            if other_key == key:
                continue

            dis_loss = self._classification_metric(discriminator_scores, disc_label_dict[other_key])
            # add each discriminator prediction equally to the per model sum
            model_dis_loss += 1 / (len(disc_label_dict) - 1) * dis_loss

        return model_dis_loss

    def _track_translation(
        self, model_key: str, forward_pass: Dict[str, List[torch.Tensor]], batch: Dict[str, Dict[str, torch.Tensor]]
    ) -> float:
        """ """
        self._set_eval_models()

        trans_loss = 0
        with torch.no_grad():
            for other_key in forward_pass.keys():
                if other_key == model_key:
                    continue

                latents = forward_pass[model_key][1].detach().clone()
                trans = self.model_dict[other_key].decode(latents)

                # potentially need to move other_data to correct device
                trans_loss += self._anchor_metric(self._batchtensor_to_device(batch[other_key]['data']), trans)

        self._set_train_models()

        return trans_loss / (len(self.model_dict) - 1)

    def _compute_anchor_loss(self, key: str, forward_pass: Dict[str, List[torch.Tensor]]) -> float:
        """ """
        curr_latents = forward_pass[key][1]
        other_latents = []

        # get other latents from forward pass dict
        for o_key, o_results in forward_pass.items():
            if o_key != key:
                other_latents.append(o_results[1])

        anchor_loss = self._anchor_metric(
            curr_latents.repeat((1, len(forward_pass) - 1)),  # repeat current latent to number of other models
            torch.cat(other_latents, dim=1),  # concatenate other latents to large tensor
        )

        return anchor_loss

    def _train_discriminator_epoch(
        self, forward_pass: Dict[str, List[torch.Tensor]], disc_label_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """ """
        # set discriminator to train
        self.discriminator.train()
        # zero_grad discriminator optimizer
        self.discriminator.zero_grad()

        dis_loss_dict = {}
        # loop through per model forward pass results
        for key, forward_result in forward_pass.items():
            latents = forward_result[1]

            # forward pass on discriminator
            # TODO: add option to use cluster labels in discriminator
            discriminator_scores = self.discriminator(latents.detach().clone())

            # compute loss for correct class label and add to loss dict
            dis_loss_dict[key] = self._classification_metric(discriminator_scores, disc_label_dict[key])

        return dis_loss_dict

    def _val(self, data_dict: Dict[str, np.array], batch_size: int = 256) -> Dict[str, AverageMeter]:
        # set up a data loader from the input data
        val_dataloader = self._set_up_dataloader(data_dict, batch_size, shuffle=False)

        # set all auto encoders to eval
        self._set_eval_models()

        norm_factor = 1 / len(self.model_dict)

        val_meter_dict: Dict[str, AverageMeter] = {}
        for _idx, batch in enumerate(val_dataloader):
            # run all models without tracking gradients
            with torch.no_grad():
                # run forward pass
                forward_pass = self._ae_forward_pass(batch)

                # get fake discrimnator labels
                disc_label_dict = self._build_disc_label_dict(batch)

                model_loss_dict, classifier_loss_dict = self._compute_model_loss(
                    batch, forward_pass, disc_label_dict, val_meter_dict
                )

            # build weighted sum loss over all individual models
            loss = sum(norm_factor * value for value in model_loss_dict.values())

            # optional: update classifier
            if self.use_classifier:
                # log classifier loss
                classifier_loss = sum(norm_factor * value for value in classifier_loss_dict.values())
                self._update_meter_dict(val_meter_dict, 'classifier_loss', classifier_loss)
                loss += classifier_loss

            # check that the loss is not na
            assert not torch.isnan(loss)
            self._update_meter_dict(val_meter_dict, 'loss', loss)

            for model_key in data_dict.keys():
                # translation loss (for tracking)
                trans_loss = self._track_translation(model_key, forward_pass, batch)
                self._update_meter_dict(val_meter_dict, f'{model_key}_translation_loss', trans_loss)

        return val_meter_dict

    def _step_autoencoders(self):
        for _key, model in self.model_dict.items():
            # self.model_dict[key] = model.step()

            # check if the model should stay frozen during joint training
            if model.train_joint:
                model.step()

    def _zero_grad_autoencoders(self):
        for _key, model in self.model_dict.items():
            # self.model_dict[key] = model.zero_grad()
            model.zero_grad()

    def _set_train_models(self):
        for key, model in self.model_dict.items():
            # check if the model should stay frozen during joint training
            if model.pretrain_epochs == 0 and not model.train_joint:
                self.model_dict[key] = model.eval()
            else:
                self.model_dict[key] = model.train()

    def _set_eval_models(self):
        for key, model in self.model_dict.items():
            self.model_dict[key] = model.eval()

    def _to_device(self, obj: Any, device: str):
        if device == 'gpu':
            obj.cuda()
        else:
            obj.detach().cpu()

    def forward(self, model_key: str, X: np.array, batch_size: int = 256, use_gpu: bool = False) -> tuple:
        """
        Utility function to do a forward run of a numpy array on a single model, with batching and gpu utilization.

        model_key: modality name/id that was also used during training
        X: numpy array to be run forward through the specified model

        return:
        recon_ar: numpy array of reconstructed data
        latent_ar: numpy array of latent reptresentation of X
        """
        # set up simple data loader from the input data
        dataloader = self._dataloader_from_numpy(X, batch_size, False)

        # get correct model
        model = self.model_dict[model_key]
        # set model to evaluation mode
        model.eval()

        # move model to GPU if requested
        if use_gpu:
            model.cuda()

        recon_ar_list = []
        latents_ar_list = []

        # iterate through batches of input data
        for _idx, batch in enumerate(dataloader):
            data = batch[0]

            # don't track gradients
            with torch.no_grad():
                recon, latents, _, _ = model.forward(self._batchtensor_to_device(data))

            recon_ar_list.append(recon)
            latents_ar_list.append(latents)

        # build full arrays from batched data
        recon_ar = torch.cat(recon_ar_list, dim=0)
        latents_ar = torch.cat(latents_ar_list, dim=0)

        if use_gpu:
            recon_ar.detach().cpu()
            latents_ar.detach().cpu()

        return recon_ar.cpu().numpy(), latents_ar.cpu().numpy()

    def translate(
        self, from_key: str, to_key: str, from_X: np.array, batch_size: int = 256, use_gpu: bool = False
    ) -> np.array:
        """
        Utility function allowing to translate a numpy array between any two registered modalities/models.
        """
        # set up dataloader
        dataloader = self._dataloader_from_numpy(from_X, batch_size, False)

        from_model = self.model_dict[from_key]
        to_model = self.model_dict[to_key]

        from_model.eval()
        to_model.eval()

        if use_gpu:
            from_model.cuda()
            to_model.cuda()

        trans_ar_list = []
        # iterate through batches of input data
        for _idx, batch in enumerate(dataloader):
            # need this workaround as the default dataloader returns a list
            data = batch[0]

            if use_gpu:
                data.cuda()

            # don't track gradients
            with torch.no_grad():
                z = from_model.encode(self._batchtensor_to_device(data))
                trans = to_model.decode(z)

            trans_ar_list.append(trans)

        # build full arrays from batched data
        trans_ar = torch.cat(trans_ar_list, dim=0)

        if use_gpu:
            trans_ar.detach().cpu()

        return trans_ar.cpu().numpy()

    def _dataloader_from_numpy(self, X: np.array, batch_size: int, shuffle: bool):
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X.copy()).float()),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

        return dataloader

    def _batchtensor_to_device(self, data: torch.Tensor) -> torch.Tensor:
        """Utility function to move tensor from batch dict as Variable to device.

        Args:
            data (torch.Tensor): torch Tensor coming form the batch dictionary

        Returns:
            torch.Tensor: Variable version of the input moved to device.
        """
        X = Variable(data)
        X = X.cuda() if self.device == 'cuda' else X

        return X

    def _update_meter(self, meter_name: str, value: torch.Tensor):
        if meter_name in self.meter_dict.keys():
            tmp_meter = self.meter_dict[meter_name]
        else:
            tmp_meter = AverageMeter(meter_name)

        # check if its a simple scaler or a tensor
        # if its a tensor only get the item value
        # to avoid unwanted graph tracking
        v = value.item() if isinstance(value, torch.Tensor) else value
        tmp_meter.update(v)

        self.meter_dict[meter_name] = tmp_meter

    def _update_meter_dict(self, meter_dict: Dict[str, AverageMeter], meter_name, value: torch.Tensor):
        if meter_name in meter_dict.keys():
            tmp_meter = meter_dict[meter_name]
        else:
            tmp_meter = AverageMeter(meter_name)

        # check if its a simple scaler or a tensor
        # if its a tensor only get the item value
        # to avoid unwanted graph tracking
        v = value.item() if isinstance(value, torch.Tensor) else value
        tmp_meter.update(v)

        meter_dict[meter_name] = tmp_meter

    def _write_meter_log(self, epoch: int, log_path: str = ''):
        """
        Probably better at some point as stand alone function in util

        Go through all registered meters and write to file.
        """

        # build nice string
        log_entry = [f'epoch:{epoch}'] + [f'{meter.name}:{meter.val}' for meter in self.meter_dict.values()]

        if len(log_path) > 0:
            # check if log path is a valid path
            # write nice string to file

            log_string = ','.join(log_entry) + '\n'
            with open(log_path, 'a') as f:
                print(log_string, file=f)
        else:
            pprint.pprint(log_entry)

    def _safe_checkpoint(self, key: str, model: Union[VariationalAutoencoder, Classifier], checkpoint_dir: str) -> None:
        """
        Better at some point as stand alone function in util

        Store model and optimizer state dict to checkpoint_dir
        """

        # create directory to store state dicts in
        ckp_dir = os.path.join(checkpoint_dir, key)

        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)

        # store autoencoder and according optimizer to the same file
        if key == 'discriminator':
            torch.save(self.discriminator.cpu().state_dict(), os.path.join(ckp_dir, 'discriminator.pth'))
        elif key == 'classifier':
            torch.save(self.classifier.cpu().state_dict(), os.path.join(ckp_dir, 'classifier.pth'))
        else:
            torch.save(
                {'autoencoder': model.cpu().state_dict(), 'optimizer': model._optimizer.state_dict()},
                os.path.join(ckp_dir, 'joint_vae.pth'),
            )

    def save_model(self, checkpoint_dir: str):
        """
        Public method to save all parts of the joint model
        """
        readme_list = []
        for key, model in self.model_dict.items():
            self._safe_checkpoint(key, model, checkpoint_dir)

            readme_list.append(f'-- Modality key: {key} --')
            readme_list.append(str(model))

        self._safe_checkpoint('discriminator', self.discriminator, checkpoint_dir)
        readme_list.append('-- Discriminator -')
        readme_list.append(str(self.discriminator))

        if self.use_classifier:
            self._safe_checkpoint('classifier', self.classifier, checkpoint_dir)
            readme_list.append('-- Classifier --')
            readme_list.append(str(self.classifier))

        # write some meta information to README file
        with open(os.path.join(checkpoint_dir, 'README.txt'), 'w') as f:
            print('\n'.join(readme_list), file=f)

        # store loss weights
        torch.save(
            {
                'recon_weight': self.recon_weight_dict,
                'beta': self.beta,
                'cl_weight': self.cl_weight,
                'disc_weight': self.disc_weight,
                'anchor_weight': self.anchor_weight,
                'paired': self.paired,
            },
            os.path.join(checkpoint_dir, 'loss_weights.pth'),
        )

        # move models to appropriate device in case we continue
        # training after this method call
        self._dict_to_device(self.model_dict)
        self.discriminator = self.discriminator.to(self.device)

        # check classifier status
        if self.use_classifier:
            self.classifier.to(self.device)

    # TODO: integrate into pretraining
    def frange_cycle_linear(
        self,
        start: float,
        stop: float,
        n_epoch: int,
        n_cycle: int = 1,
        ratio: float = 0.5,
        rep: int = 1,
        warm_up: int = 20,
    ) -> np.array:
        """Cyclic annealing function for importance of KL loss for AE pretraining.

        Args:
            start (float): Beta value to start from
            stop (float): Beta value to stop at
            n_epoch (int): Number of epochs over which the annealing should happen
            n_cycle (int, optional): Number of cylces to use in annealing. Defaults to 1.
            ratio (float, optional):
                Ratio of epochs to spend on annealing compared to staying at beta stop value. Defaults to 0.5.
            rep (int, optional): Number of times to repeat the same beta step. Defaults to 1.
            warm_up (int, optional): Number of epochs to train before any variational loss is applied. Defaults to 20.

        Returns:
            np.array (float): Returns a numpy array with n_epoch beta values.
        """

        n_step_epochs = int((n_epoch - warm_up) / n_cycle)  # how many epochs do we have per cycle
        step_epochs = int(n_step_epochs * ratio)  # how many epochs should be used for increasing beta
        n_max_epochs = n_step_epochs - step_epochs  # number of epochs we need to stay at max_beta for
        step_epochs = int(step_epochs / rep)  # account for number of repeated step values

        # create the linear steps for one period
        step_array, step_size = np.linspace(start, stop, step_epochs, retstep=True)
        print(f'Using a step size of {step_size}')

        # check if steps need to stay at same value for multiple epochs
        if rep > 1:
            # built step array with repeated values
            step_array = np.repeat(step_array, rep)

        # construct the array for one period
        period_array = np.append(step_array, np.repeat(stop, n_max_epochs))
        # add as many period arrays as specified and concat to final array with warm up
        beta_array = np.concatenate([np.zeros(warm_up), np.tile(period_array, n_cycle)])

        # account for rounding issues and add a couple more steps
        # to warm up and to late stage training
        epoch_diff = n_epoch - beta_array.size
        if epoch_diff > 0:
            warmup_add = int(np.ceil(epoch_diff / 2))
            maxval_add = int(np.floor(epoch_diff / 2))
            print(f'Adding {warmup_add} epochs to warm-up and {maxval_add} end max value')

            beta_array = np.append(np.zeros(warmup_add), beta_array)
            beta_array = np.append(beta_array, np.repeat(stop, maxval_add))

        return beta_array
