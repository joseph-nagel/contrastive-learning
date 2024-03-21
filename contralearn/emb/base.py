'''Embedding model.'''

import torch
from lightning.pytorch import LightningModule

from ..loss import OnlineTripletLoss


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
    squared : bools
        Determines whether the Euclidean distance is squared.
    eps : float
        Small epsilon to avoid zeros.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(self,
                 embedding,
                 margin,
                 mine_mode='batch_all',
                 squared=True,
                 eps=1e-06,
                 lr=1e-04):

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

    def __call__(self, x):
        '''Embed the inputs.'''
        return self.embedding(x)

    @staticmethod
    def _get_batch(batch):
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

    def loss(self, x, y):
        '''Compute the triplet loss.'''
        embeddings = self(x)
        loss = self.triplet_loss(embeddings, y)
        return loss

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch) # note that Lightning dismisses steps with None loss
        if loss is not None:
            self.log('train_loss', loss.item()) # Lightning logs batch-wise metrics during training per default
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        if loss is not None:
            self.log('val_loss', loss.item()) # Lightning automatically averages metrics over batches for validation
        return loss

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        if loss is not None:
            self.log('test_loss', loss.item()) # Lightning automatically averages metrics over batches for testing
        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

