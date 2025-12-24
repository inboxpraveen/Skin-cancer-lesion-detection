"""
Setup script for Skin Cancer Detection System
"""

import os
from setuptools import setup, find_packages

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='skin-cancer-detection',
    version='1.0.0',
    author='Skin Cancer Detection Team',
    description='Production-grade deep learning system for automated skin lesion classification',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/inboxpraveen/skin-cancer-detection',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.10',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'skin-cancer-train=src.train:main',
            'skin-cancer-evaluate=src.evaluate:main',
            'skin-cancer-predict=src.inference:main',
            'skin-cancer-camera=src.camera_service:main',
        ],
    },
)

