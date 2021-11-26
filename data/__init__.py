""" Data package is in charge of
    a) splitting the data (i.e creating the problem setup) as
    b) as well as returning data loader that let's you sample from the data
"""

from . import femnist_dataset
from . import cifar_dataset
from .loader import get_loaders
