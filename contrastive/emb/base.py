'''Embedding model.'''

import torch
from lightning.pytorch import LightningModule


class Embedding(LightningModule):
    '''
    Base embedding model.

    Parameters
    ----------
    embedding : PyTorch module
        Model implementing the embedding.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(self, embedding, lr=1e-04):
        super().__init__()

        # set embedding model
        self.embedding = embedding

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

    def loss(self, x):
        '''Compute the loss.'''
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        self.log('train_loss', loss.item()) # Lightning logs batch-wise metrics during training per default
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        self.log('val_loss', loss.item()) # Lightning automatically averages metrics over batches for validation
        return loss

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)
        self.log('test_loss', loss.item()) # Lightning automatically averages metrics over batches for testing
        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

