'''
Embedding model.

Modules
-------
base : Base embedding.
conv : Conv. embedding.
utils : Some utilities.

'''

from . import base, conv, utils
from .base import Embedding
from .conv import ConvEmbedding
from .utils import embed_loader
