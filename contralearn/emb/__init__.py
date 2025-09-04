'''
Embedding model.

Modules
-------
base : Base embedding.
conv : Conv. embedding.
utils : Some utilities.

'''

from . import base
from . import conv
from . import utils


from .base import Embedding

from .conv import ConvEmbedding

from .utils import embed_loader
