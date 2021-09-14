# -*- coding: utf-8 -*-


from setuptools import setup

setup(
    name='panopticapi',
    packages=['panopticapi'],
    package_dir={'panopticapi': 'panopticapi'},
    install_requires=[
        'numpy',
        'Pillow',
    ],
    version='0.1',
)
