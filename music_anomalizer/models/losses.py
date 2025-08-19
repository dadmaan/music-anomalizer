import torch
import torch.nn.functional as F


def clip_bce(output, target):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy(
        output, target)


def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce