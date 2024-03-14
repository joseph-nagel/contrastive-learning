'''Some utilities.'''

import torch


@torch.no_grad()
def embed_loader(emb, data_loader, return_labels=False):
    '''Embed all items in a data loader.'''

    emb.train(False) # activate train mode

    embeddings = []

    if return_labels:
        labels = []

    # loop over batches
    for x_batch, y_batch in data_loader:

        if return_labels:
            labels.append(y_batch)

        # compute embedding
        e_batch = emb(x_batch.to(emb.device))
        e_batch = e_batch.cpu()

        embeddings.append(e_batch)

    embeddings = torch.cat(embeddings, dim=0)

    if return_labels:
        labels = torch.cat(labels, dim=0)

    if return_labels:
        return embeddings, labels
    else:
        return embeddings

