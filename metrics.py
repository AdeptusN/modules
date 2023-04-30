'''Custom metrics by AdeptusN'''

import torch
import torch.nn as nn
import torch.nn.functional as functional


class IoUMetricBin(nn.Module):
    """
    Metric for binary segmentation

    It is calculated as the ratio between the overlap of
    the positive instances between two sets, and their mutual combined values

    J(A, B) = |A and B| / |A or B| = |A and B| / (|A| + |B| - |A and B|)

    """
    def __init__(self):
        super(IoUMetricBin, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU


class DiceScoreBin(nn.Module):
    """
    Metric for binary segmentation

    DSC(A, B) = 2 |A and B| / (|A| + |B|)

    """
    def __init__(self, smooth=1):
        super(DiceLossBin, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2 * intersection + self.smooth) / (inputs.sum() + targets.sum())

        return dice