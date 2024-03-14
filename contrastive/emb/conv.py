'''Conv. embedding.'''

import torch.nn as nn

from .base import Embedding


class ConvEmbedding(Embedding):
    '''
    Conv. embedding model.

    Parameters
    ----------
    num_channels : list
        Number of hidden channels.
    embed_dim : int
        Dimension of the embedding.
    margin : float
        Margin of the triplet loss.
    squared : bools
        Determines whether the Euclidean distance is squared.
    eps : float
        Small epsilon to avoid zeros.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(self,
                 num_channels,
                 embed_dim,
                 margin,
                 squared=True,
                 eps=1e-06,
                 lr=1e-04):

        # check channel numbers
        if len(num_channels) != 2:
            raise ValueError('Exactly two channel numbers expected')

        # create embedding model
        conv_layers = [
            nn.Conv2d(1, num_channels[0], 3, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(num_channels[0], num_channels[1], 3, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
        ]

        dense_layer = nn.Linear(num_channels[1] * 7 * 7 , embed_dim)

        embedding = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            dense_layer
        )

        # initialize embedding class
        super().__init__(
            embedding,
            margin,
            squared=squared,
            eps=eps,
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

