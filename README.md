# ChemBioMultimodalAutoencoders
 

<div>

 |||
| --- | --- |
| CI/CD | [![tests](https://github.com/Novartis/ChemBioMultimodalAutoencoders/actions/workflows/python-package-tests.yml/badge.svg?branch=main)](https://github.com/Novartis/ChemBioMultimodalAutoencoders/actions/workflows/python-package-tests.yml) [![builds](https://github.com/Novartis/ChemBioMultimodalAutoencoders/actions/workflows/build-and-publish.yml/badge.svg)](https://github.com/Novartis/ChemBioMultimodalAutoencoders/actions/workflows/build-and-publish.yml) |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/multimodal-autoencoders.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/multimodal-autoencoders/) ![Downloads](https://static.pepy.tech/badge/multimodal-autoencoders)[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/multimodal-autoencoders.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/multimodal-autoencoders/) |
| Meta | [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) [![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![imports - isort](https://img.shields.io/badge/imports-isort-ef8336.svg)](https://github.com/pycqa/isort) [![LICENSE](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Novartis/ChemBioMultimodalAutoencoders/blob/readme-update/license.txt)

 

</div>

a package for streamlined multidomain data integration and translation based on [cross-modal autoencoder architecture](https://github.com/uhlerlab/cross-modal-autoencoders)[[1]](https://github.com/Novartis/ChemBioMultimodalAutoencoders#references). It is designed to add new data modalities and train models for seamless translation. 

# Installation
To install the package, simply run:

```pip install multimodal-autoencoders```

**(optional)**
To make sure you have all the dependencies, you can create an appropriate environment using the environment.yml file with Conda:

```conda env create -f environment.yml```

# Usage
An example on how to train and use the multimodal autoencoders can be found in relevant notebooks in `examples` <br>
<br>
Usage is centered around a JointTraner instance (defined in multimodal_autoencoders/trainer/joint_trainer.py). A central part of the whole architecture is that different components need to be associated to the individual modalities. This is done through python dictionaries, with which most users will be familiar with.<br>

# Authors
**Thibault Bechtler** (th.bechtler@gmail.com) & **Bartosz Baranowski** (bartosz.baranowski@novartis.com)

Contributors:
**Michal Pikusa** (michal.pikusa@novartis.com), **Steffen Renner** (steffen.renner@novartis.com)

# References
```[1] Yang, K.D., Belyaeva, A., Venkatachalapathy, S. et al. Multi-domain translation between single-cell imaging and sequencing data using autoencoders. Nat Commun 12, 31 (2021). https://doi.org/10.1038/s41467-020-20249-2```

