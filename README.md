# ChemBioMultimodalAutoencoders
a package for streamlined multidomain data integration and translation based on [cross-modal autoencoder architecture](https://github.com/uhlerlab/cross-modal-autoencoders). It is designed to add new data modalities and train models for seamless translation. 

# Setting up the environment
You can create an appropriate environment using the environment.yml file with Conda:

```conda env create -f environment.yml```

# Usage
An example on how to train and use the multimodal autoencoders can be found in relevant notebooks in `examples` <br>
<br>
Usage is centered around a JointTraner instance (defined in multimodal_autoencoders/trainer/joint_trainer.py). A central part of the whole architecture is that different components need to be associated to the individual modalities. This is done through python dictionaries, with which most users will be familiar with.<br>

# Authors
Thibault Bechtler & [Bartosz Baranowski](bartosz.baranowski@novartis.com)

Contributors: Michal Pikusa, Steffen Renner
