#-*- coding: utf-8 -*-


from setuptools import setup, Extension

setup(
    name='panopticapi',
    packages=['panopticapi', 'panopticapi_converters'],
    package_dir = {'panopticapi': 'panopticapi',
                   'panopticapi_converters': 'converters'},
    install_requires=[
        'numpy',
        'Pillow',
    ],
    version='0.1',
)
