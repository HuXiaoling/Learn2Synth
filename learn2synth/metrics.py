from torch import nn
import torch
from cornucopia.utils.morpho import erode
from distmap import euclidean_distance_transform
from . import utils


def _dot(x, y):
    """Dot product along the last dimension"""
    return x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1).squeeze(-1)


class Metric(nn.Module):
    """Base class for non differentiable metrics"""

    def __init__(self, reduction='mean'):
        """
        Parameters
        ----------
        reduction : {'mean', 'sum'} or callable
            Reduction to apply across batch elements
        """
        super().__init__()
        self.reduction = reduction

    def reduce(self, x):
        if not self.reduction:
            return x
        if isinstance(self.reduction, str):
            if self.reduction.lower() == 'mean':
                return x.mean()
            if self.reduction.lower() == 'sum':
                return x.sum()
            raise ValueError(f'Unknown reduction "{self.reduction}"')
        if callable(self.reduction):
            return self.reduction(x)
        raise ValueError(f'Don\'t know what to do with reduction: '
                         f'{self.reduction}')


class Dice(Metric):
    r"""Hard Dice

    By default, each class is weighted identically.
    The `weighted` mode allows classes to be weighted by frequency.
    """

    def __init__(self, weighted=False, labels=None, reduction='mean',
                 exclude_background=True, background=0):
        """

        Parameters
        ----------
        weighted : bool or list[float], default=False
            If True, weight the Dice of each class by its frequency in the
            reference. If a list, use these weights for each class.
        labels : list[int], default=range(nb_class)
            Label corresponding to each one-hot class. Only used if the
            reference is an integer label map.
        reduction : {'mean', 'sum', None} or callable, default='mean'
            Type of reduction to apply across minibatch elements.
        exclude_background : bool
            Exclude background class from Dice
        background : int
            Index of the background class.
        """
        super().__init__(reduction)
        self.weighted = weighted
        self.labels = labels
        self.exclude_background = exclude_background
        self.background = background

    def forward(self, pred, ref, mask=None):
        """

        Parameters
        ----------
        pred : (batch, nb_class, *spatial) tensor
            Predicted classes.
        ref : (batch, nb_class|1, *spatial) tensor
            Reference classes (or their expectation).
        mask : (batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or (batch,) tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar tensor.

        """
        nb_classes = pred.shape[1]
        backend = dict(dtype=pred.dtype, device=pred.device)

        # convert prob/onehot to labels
        pred = pred.argmax(1, keepdim=True)
        if ref.dtype.is_floating_point:
            ref = ref.argmax(1, keepdim=True)

        # prepare weights
        weights = self.weighted
        if not torch.is_tensor(weights) and not weights:
            weights = False
        if not isinstance(weights, bool):
            weights = utils.make_vector(weights, nb_classes, **backend)

        labels = self.labels or list(range(nb_classes))

        loss = 0
        sumweights = 0
        for index, label in enumerate(labels):
            if label is None:
                continue
            if self.exclude_background and index == self.background:
                continue
            pred1 = (pred == index).squeeze(1)
            ref1 = (ref == label).squeeze(1)
            if mask is not None:
                pred1 = pred1 * mask
                ref1 = ref1 * mask

            pred1 = pred1.reshape([len(pred1), -1])           # [B, N]
            ref1 = ref1.reshape([len(ref1), -1])              # [B, N]
            hasref = ref1.any(1)

            # Compute Dice
            inter = (pred1 * ref1).sum(-1)                    # [B]
            pred1 = pred1.sum(-1)                             # [B]
            ref1 = ref1.sum(-1)                               # [B]
            union = pred1 + ref1
            loss1 = (2 * inter) / union
            loss1.masked_fill_(~hasref, 0)

            # Simple or weighted average
            if weights is not False:
                if weights is True:
                    weight1 = ref1
                else:
                    weight1 = float(weights[index]) * hasref
                loss1 = loss1 * weight1
                sumweights += weight1
            else:
                sumweights += hasref
            loss += loss1

        # Minibatch reduction
        loss = loss / sumweights
        return self.reduce(loss)


class Hausdorff(Metric):
    r"""Hausdorff distance

    By default, each class is weighted identically.
    The `weighted` mode allows classes to be weighted by frequency.
    """

    def __init__(self, weighted=False, pct=1., directed=True, labels=None,
                 exclude_background=True, background=0, voxel_size=1.,
                 reduction='mean'):
        """

        Parameters
        ----------
        weighted : bool or list[float], default=False
            If True, weight the Dice of each class by its frequency in the
            reference. If a list, use these weights for each class.
        pct : float
            Distance quantile
        directed : bool
            Compute the directed distance
        labels : list[int], default=range(nb_class)
            Label corresponding to each one-hot class. Only used if the
            reference is an integer label map.
        exclude_background : bool
            Exclude background class from Dice
        background : int
            Index of the background class.
        voxel_size : [sequence of] float
            Voxel size
        reduction : {'mean', 'sum', None} or callable, default='mean'
            Type of reduction to apply across minibatch elements.
        """
        super().__init__(reduction)
        self.weighted = weighted
        self.labels = labels
        self.exclude_background = exclude_background
        self.background = background
        self.voxel_size = voxel_size
        self.directed = directed

    def forward(self, pred, ref):
        """

        Parameters
        ----------
        pred : (batch, nb_class, *spatial) tensor
            Predicted classes.
        ref : (batch, nb_class|1, *spatial) tensor
            Reference classes (or their expectation).
        mask : (batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or (batch,) tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar tensor.

        """
        nb_classes = pred.shape[1]
        backend = dict(dtype=pred.dtype, device=pred.device)

        # convert prob/onehot to labels
        pred = pred.argmax(1)
        if ref.dtype.is_floating_point:
            ref = ref.argmax(1)

        # prepare weights
        weights = self.weighted
        if not torch.is_tensor(weights) and not weights:
            weights = False
        if not isinstance(weights, bool):
            weights = utils.make_vector(weights, nb_classes, **backend)

        labels = self.labels or list(range(nb_classes))

        loss = 0
        sumweights = 0
        for index, label in enumerate(labels):
            if label is None:
                continue
            if self.exclude_background and index == self.background:
                continue
            pred1 = (pred == index)
            ref1 = (ref == label)

            # Compute distance
            loss1 = []
            for pred11, ref11 in zip(pred1, ref1):
                loss1 += [hausdorff(pred11, ref11, directed=self.directed,
                                    vx=self.voxel_size)]
            loss1 = torch.stack(loss1)

            # Mask missing labels
            ref1 = ref1.reshape([len(ref1), -1])              # [B, N]
            hasref = ref1.any(1)
            loss1.masked_fill_(~hasref, 0)

            # Simple or weighted average
            if weights is not False:
                if weights is True:
                    weight1 = ref1.sum(-1)
                else:
                    weight1 = float(weights[index]) * hasref
                loss1 = loss1 * weight1
                sumweights += weight1
            else:
                sumweights += hasref
            loss += loss1

        # Minibatch reduction
        loss = loss / sumweights
        return self.reduce(loss)


def get_border(mask):
    """Compute mask of the inner border of a mask

    Parameters
    ----------
    mask : (*shape) tensor
        Input mask

    Returns
    -------
    border : (*hape) tensor
        Border mask

    """
    border = mask ^ erode(mask, dim=mask.ndim)
    return border


def get_surface_distance(border_pred, border_ref, vx=1.):
    """Compute the distance to border_ref at each point of border_pred

    Parameters
    ----------
    border_pred : (*shape) tensor
        Mask of the border points of the prediction
    border_ref : (*shape) tensor
        Mask of the border points of the reference
    vx : (ndim,) sequence[float]
        Voxel size

    Returns
    -------
    dist : (nb_points,) tensor
        Euclidean distance to the reference border at each point of
        the predicted border

    """
    ndim = border_pred.ndim
    dist = euclidean_distance_transform(border_ref, ndim=ndim, vx=vx)
    return dist[border_pred]


def hausdorff(mask_pred, mask_ref, directed=True, pct=1., vx=1.):
    """Compute the Hasudorff distance between two segmentations

    Parameters
    ----------
    mask_pred : (*shape) tensor
        Predicted mask
    mask_ref : (*shape) tensor
        Reference mask
    directed : bool
        Compute the directed distance
    pct : float
        Distance percentile
    vx : [sequence] float
        Voxel size

    Returns
    -------
    dist : scalar tensor
        Hausdorff distance

    """
    border_pred = get_border(mask_pred)
    border_ref = get_border(mask_ref)
    dist = get_surface_distance(border_pred, border_ref, vx=vx)
    dist = torch.quantile(dist, pct) if pct < 1 else dist.max()
    if not directed:
        dist2 = get_surface_distance(border_ref, border_pred, vx=vx)
        dist2 = torch.quantile(dist2, pct) if pct < 1 else dist2.max()
        dist = max(dist, dist2)
    return dist
