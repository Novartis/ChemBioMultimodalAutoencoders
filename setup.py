from setuptools import setup

setup(
    name='multimodal_autoencoders',
    version='0.1',
    author='Bechtler Thibault',
    author_email='thibault.bechtler@novartis.com',
    packages=['multimodal_autoencoders'],
    long_description=open("README.md").read(),
    install_requires=[
        "dataclasses",
        "numpy",
        "torch",
        "scikit-learn",
    ],
)
