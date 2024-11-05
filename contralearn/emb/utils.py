'''Some utilities.'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def embed_loader(
    emb: nn.Module,
    data_loader: DataLoader,
    return_labels: bool = False
):
    '''Embed all items in a data loader.'''

    emb.train(False) # activate train mode

    embeddings_list = []

    if return_labels:
        labels_list = []

    # loop over batches
    for x_batch, y_batch in data_loader:

        if return_labels:
            labels_list.append(y_batch)

        # compute embedding
        e_batch = emb(x_batch.to(emb.device))
        e_batch = e_batch.cpu()

        embeddings_list.append(e_batch)

    embeddings = torch.cat(embeddings_list, dim=0)

    if return_labels:
        labels = torch.cat(labels_list, dim=0)

    if return_labels:
        return embeddings, labels
    else:
        return embeddings

