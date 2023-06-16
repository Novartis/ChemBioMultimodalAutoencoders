# ChemBioMultimodalAutoencoders
a package for streamlined multidomain data integration and translation based on cross-modal autoencoders architectures.

# Setting up the environment
You can create an appropriate environment using the envirnment.yml file with conda:
```conda env create -f environment.yml```

# Basic Usage
An example on how to train and use the multimodal autoencoders can be found in `examples/basic_usage.ipynb`<br>
<br>
Usage is centered around a JointTraner instance (defined in multimodal_autoencoders/trainer/joint_trainer.py). A central part of the whole architecture is that different components need to be associated to the individual modalities. This is done through python dictionaries, with which most users will be familiar.<br>

## Essential Components
There are three essential components to the multimodal architecture:
- A model dictionary containing string - autoencoder pairs.
- A data dictionary containing string - numpy array pairs.
- A discriminator instance.
  
**Important: all string keys of the dictionaries need to match across all dictionaries!**

### Model Dictionary
The model dictionary should be a Python dictionary mapping a string name to an autoencoder. This can be an instance of the already defined VariationalAutoencoder class, or your own implementation. Please make sure that your own implementation is a subclass of the base autoencoder (definded in multimodal_autoencoders/base/autoencoder.py). Implementing the abstract functions ensures a seemles integration into the JointTrainer.

### Data Dictionary
The data dictionary should be a Python dictionary mapping a string name to a numpy array. The names of the data dicionary entries need to match with the appropriate model in the model dictionary. <br>
Disclaimer: At the moment only paired data sets are supported! All preprocessing and matching of the data is your responsiblity!

## Optional Components
### Classifier
If available, the architecture can also condition the models take known clusters into account. For that you need to pass a classifier instance to the JointTrainer.

### Cluster Labels
The cluster labels need to be provided as a numpy array. The order of values inside the array needs to correspond to the order in the data provided in the data dictionary for now. This will change for future versions supporting unpaired data.
