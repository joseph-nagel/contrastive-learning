'''
Contrastive learning.

Modules
-------
data : Datamodules.
emb : Embedding model.
loss : Loss functions.

'''

from . import data
from . import emb
from . import loss


from .data import MNISTDataModule

from .emb import (
    Embedding,
    ConvEmbedding,
    embed_loader
)

from .loss import (
    pairwise_distances,
    make_all_triplet_ids,
    OnlineTripletLoss
)

