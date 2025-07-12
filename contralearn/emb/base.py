'''Base embedding.'''

from collections.abc import Sequence

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from ..loss import OnlineTripletLoss


# define type alias
BatchType = torch.Tensor | Sequence[torch.Tensor] | dict[str, torch.Tensor]


class Embedding(LightningModule):
    '''
    Base embedding model.

    Parameters
    ----------
    embedding : PyTorch module
        Model implementing the embedding.
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
        embedding: nn.Module,
        margin: float,
        mine_mode: str = 'batch_all',
        squared: bool = True,
        eps: float = 1e-06,
        lr: float = 1e-04
    ) -> None:

        super().__init__()

        # set embedding model
        self.embedding = embedding

        # set loss function
        self.triplet_loss = OnlineTripletLoss(
            margin=margin,
            mine_mode=mine_mode,
            squared=squared,
            eps=eps
        )

        # set initial learning rate
        self.lr = abs(lr)

        # store hyperparams
        self.save_hyperparameters(
            ignore=['embedding'],
            logger=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Embed the inputs.'''
        return self.embedding(x)

    @staticmethod
    def _get_batch(batch: BatchType) -> tuple[torch.Tensor, torch.Tensor]:
        '''Get batch features and labels.'''

        if isinstance(batch, (tuple, list)):
            x_batch = batch[0]
            y_batch = batch[1]

        elif isinstance(batch, dict):
            x_batch = batch['features']
            y_batch = batch['labels']

        else:
            raise TypeError(f'Invalid batch type: {type(batch)}')

        return x_batch, y_batch

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''Compute the triplet loss.'''
        embeddings = self(x)
        loss = self.triplet_loss(embeddings, y)
        return loss

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)  # note that Lightning dismisses steps with None loss
        if loss is not None:
            self.log('train_loss', loss.item())  # Lightning logs batch-wise scalars during training per default
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        if loss is not None:
            self.log('val_loss', loss.item())  # Lightning automatically averages scalars over batches for validation
        return loss

    def test_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        if loss is not None:
            self.log('test_loss', loss.item())  # Lightning automatically averages scalars over batches for testing
        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

