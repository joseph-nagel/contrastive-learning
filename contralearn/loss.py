'''Loss functions.'''

import torch
import torch.nn as nn


def pairwise_distances(
    x: torch.Tensor,
    squared: bool = True,
    eps: float = 1e-06
) -> torch.Tensor:
    '''
    Compute the matrix of pairwise distances.

    Summary
    -------
    This function computes the distances between all pairs of row-vectors.
    The input is a (batch_size, feature_dim)-shaped tensor,
    while the output is (batch_size, batch_size)-shaped.

    '''

    # check tensor dimensions
    if x.ndim != 2:
        raise ValueError('Two-dim. tensor expected')

    # compute squared distances
    dot_product = torch.matmul(x, x.T) # (batch, batch)

    squared_norm = torch.diag(dot_product) # (batch)

    distances = squared_norm.unsqueeze(0) - 2.0 * dot_product + squared_norm.unsqueeze(1) # (batch, batch)

    if squared:
        distances = nn.functional.relu(distances) # avoid negative values
    else:
        distances = distances.clamp(min=eps) # avoid zero values
        distances = distances.sqrt() # compute square root

    return distances


@torch.no_grad()
def _make_pair_idxmasks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Create index mask for valid pos./neg. pairs.

    Summary
    -------
    Index masks for valid pos./neg. pairs are constructed.
    The output are two boolean tensors pos_idxmask[a, p] and neg_idxmask[a, n]
    that determine whether (a, p) and (a, n) are the indices of valid pairs.

    '''

    # check tensor dimensions
    if labels.ndim != 1:
        raise ValueError('One-dim. tensor expected')

    # construct index masks
    batch_size = len(labels)

    same_index = torch.eye(batch_size, device=labels.device).bool() # (batch, batch)
    diff_index = ~same_index

    # construct label masks
    same_label = (labels.unsqueeze(1) == labels.unsqueeze(0)) # (batch, batch)
    diff_label = ~same_label

    # create pos./neg. masks
    pos_idxmask = diff_index & same_label
    neg_idxmask = diff_index & diff_label

    return pos_idxmask, neg_idxmask


@torch.no_grad()
def _make_triplet_idxmask(labels: torch.Tensor) -> torch.Tensor:
    '''
    Create index mask for valid triplets.

    Summary
    -------
    An index mask for valid triplets is constructed.
    A boolean tensor triplet_idxmask[a, p, n] is returned.
    It determines whether (a, p, n) are the indices of a valid triplet.

    '''

    # create pos./neg. masks
    pos_idxmask, neg_idxmask = _make_pair_idxmasks(labels) # (batch, batch)

    # create triplet mask
    pos_idxmask = pos_idxmask.unsqueeze(2) # (batch, batch, 1)
    neg_idxmask = neg_idxmask.unsqueeze(1) # (batch, 1, batch)

    triplet_idxmask = (pos_idxmask & neg_idxmask) # (batch, batch, batch)

    return triplet_idxmask


@torch.no_grad()
def make_all_triplet_ids(labels: torch.Tensor) -> torch.Tensor:
    '''
    Create valid triplet IDs.

    Summary
    -------
    This function constructs the indices of valid triplets.
    It outputs a (triplets, 3)-shaped tensor whose last dimension
    contains the indices of the anchor, positive and negative samples.

    '''

    # construct triplet mask
    triplet_idxmask = _make_triplet_idxmask(labels) # (batch, batch, batch)

    # construct all valid triplet IDs
    triplet_ids = torch.nonzero(triplet_idxmask) # (triplets, 3)

    return triplet_ids


# TODO: add batch-hard mining strategy
class OnlineTripletLoss(nn.Module):
    '''
    Triplet loss with online triplet mining.

    Summary
    -------
    A triplet loss with online triplet mining is implemented.
    Only a batch-all mining strategy is supported at the moment.

    Parameters
    ----------
    margin : float
        Margin of the triplet loss.
    mine_mode : {'batch_all', 'batch_hard'}
        Batch triplet mining strategy.
    squared : bool
        Determines whether the Euclidean distance is squared.
    eps : float
        Small epsilon to avoid zeros.

    '''

    def __init__(
        self,
        margin: float,
        mine_mode: str = 'batch_all',
        squared: bool = True,
        eps: float = 1e-06
    ) -> None:

        super().__init__()

        self.margin = abs(margin)
        self.squared = squared
        self.eps = abs(eps)

        if mine_mode in ('batch_all', 'batch_hard'):
            self.mine_mode = mine_mode
        else:
            raise ValueError(f'Invalid triplet mining mode: {mine_mode}')

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:

        # construct all triplet IDs
        triplet_ids = make_all_triplet_ids(labels) # (triplets, 3)

        if len(triplet_ids) > 0:

            # compute triplet distances
            ap_terms = (embeddings[triplet_ids[:,0]] - embeddings[triplet_ids[:,1]]).pow(2) # (triplets, features)
            an_terms = (embeddings[triplet_ids[:,0]] - embeddings[triplet_ids[:,2]]).pow(2) # (triplets, features)

            ap_distances = ap_terms.sum(dim=1) # (triplets)
            an_distances = an_terms.sum(dim=1) # (triplets)

            if not self.squared:

                # avoid zero values
                ap_distances = ap_distances.clamp(min=self.eps)
                an_distances = an_distances.clamp(min=self.eps)

                # take square root
                ap_distances = ap_distances.sqrt()
                an_distances = an_distances.sqrt()

            # compute loss (batch-all)
            if self.mine_mode == 'batch_all':
                loss_terms = nn.functional.relu(ap_distances - an_distances + self.margin) # (triplets)
                loss = loss_terms.mean()

            # compute loss (batch-hard)
            else:
                raise NotImplementedError('The only supported strategy is batch-all')

        else:
            loss = None # return None (rather than NaN) in case no valid triplet can be constructed

        return loss

