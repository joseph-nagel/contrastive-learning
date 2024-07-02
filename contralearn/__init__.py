'''
Contrastive learning.

Summary
-------
A mini library for contrastive representation learning is provided.
Currently only a triplet loss function with online batch-all mining is implemented.
Other loss functions and triplet mining schemes may be included in the future.

Modules
-------
data : Datamodules.
emb : Embedding model.
loss : Loss functions.
vis : Visualization tools.

'''

from . import (
    data,
    emb,
    loss,
    vis
)

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

from .vis import make_gif, make_emb_imgs

