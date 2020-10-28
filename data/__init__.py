""" Data package is in charge of
    a) splitting the data (i.e creating the problem setup) as
    b) as well as returning data loader that let's you sample from the data
"""

from . import static_mnist
from . import celeba_dataset
from . import femnist_dataset
from .loader import get_loaders
