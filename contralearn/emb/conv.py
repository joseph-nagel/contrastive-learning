'''Conv. embedding.'''

from collections.abc import Sequence
from numbers import Number

import torch.nn as nn

from .base import Embedding


class ConvEmbedding(Embedding):
    '''
    Conv. embedding model.

    Parameters
    ----------
    num_channels : int or list
        Number of hidden channels.
    num_features : int or list
        Number of embedding features.
    margin : float
        Margin of the triplet loss.
    mine_mode : {'batch_all', 'batch_hard'}
        Batch triplet mining strategy.
    squared : bool
        Determines whether the Euclidean distance is squared.
    eps : float
        Small epsilon to avoid zeros.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(
        self,
        num_channels: int | Sequence[int],
        num_features: int | Sequence[int],
        margin: float,
        mine_mode: str = 'batch_all',
        squared: bool = True,
        eps: float = 1e-06,
        lr: float = 1e-04
    ) -> None:

        # check channel numbers
        if isinstance(num_channels, Number):
            num_channels = [num_channels]

        if len(num_channels) not in (1, 2):
            raise ValueError('One or two channel numbers expected')

        # check feature numbers
        if isinstance(num_features, Number):
            num_features = [num_features]

        if len(num_features) not in (1, 2):
            raise ValueError('One or two feature numbers expected')

        # create double conv blocks
        conv_layers = [
            nn.Conv2d(1, num_channels[0], 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(num_channels[0], num_channels[0], 3, padding='same'),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        ]

        if len(num_channels) == 2:
            conv_layers += [
                nn.Conv2d(num_channels[0], num_channels[1], 3, padding='same'),
                nn.LeakyReLU(),
                nn.Conv2d(num_channels[1], num_channels[1], 3, padding='same'),
                nn.MaxPool2d(2),
                nn.LeakyReLU()
            ]

        # create dense layers
        flattened_size = int((28 / (2**len(num_channels)))**2)

        dense_layers = [
            nn.Linear(num_channels[-1] * flattened_size , num_features[0])
        ] # type: list[nn.Module]

        if len(num_features) == 2:
            dense_layers += [
                nn.LeakyReLU(),
                nn.Linear(num_features[0], num_features[1])
            ]

        # create embedding model
        embedding = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            *dense_layers
        )

        # initialize embedding class
        super().__init__(
            embedding,
            margin,
            mine_mode=mine_mode,
            squared=squared,
            eps=eps,
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

