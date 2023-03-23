import re

from setuptools import setup


def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)


project_name = 'multimodal_autoencoders'

setup(
    name='multimodal_autoencoders',
    version=get_property('__version__', project_name),
    author='Bechtler Thibault',
    author_email='th.bechtler@gmail.com',
    packages=['multimodal_autoencoders'],
    long_description=open('README.md').read(),
    install_requires=[
        'dataclasses',
        'numpy',
        'torch',
        'scikit-learn',
    ],
)
