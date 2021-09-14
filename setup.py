# -*- coding: utf-8 -*-
import os
from pathlib import Path
from setuptools import setup


def read(file_name):
    with open(os.path.join(Path(os.path.dirname(__file__)), file_name))\
            as _file:
        return _file.read()


long_description = read('README.md')

setup(
    name='panopticapi',
    packages=['panopticapi'],
    package_dir={'panopticapi': 'panopticapi'},
    install_requires=[
        'scikit_image',
        'numpy',
        'matplotlib',
        'cityscapesscripts',
        'Pillow',
        'pycocotools',
        'skimage'
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.1',
    url='https://github.com/cocodataset/panopticapi',
    download_url='https://github.com/cocodataset/panopticapi',
)
