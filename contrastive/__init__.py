'''Contrastive learning.'''

from . import data
from . import emb
from . import loss


from .data import MNISTDataModule

from .emb import Embedding, ConvEmbedding

from .loss import (
    pairwise_distances,
    make_triplet_ids,
    OnlineTripletLoss
)

