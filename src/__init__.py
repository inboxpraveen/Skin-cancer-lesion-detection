"""
Skin Cancer Detection System

A production-grade deep learning system for automated skin lesion classification.
"""

__version__ = '1.0.0'
__author__ = 'Skin Cancer Detection Team'

from . import config
from . import data_loader
from . import models
from . import inference

__all__ = [
    'config',
    'data_loader',
    'models',
    'inference',
]

