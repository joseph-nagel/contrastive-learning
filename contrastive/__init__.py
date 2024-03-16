'''
Contrastive learning.

Modules
-------
data : Datamodules.
emb : Embedding model.
loss : Loss functions.
utils : Some utilities.

'''

from . import data
from . import emb
from . import loss
from . import utils


from .data import MNISTDataModule

from .emb import Embedding, ConvEmbedding

from .loss import (
    pairwise_distances,
    make_all_triplet_ids,
    OnlineTripletLoss
)

from .utils import embed_loader

