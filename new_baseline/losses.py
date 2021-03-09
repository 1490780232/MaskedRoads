"""PyTorch-compatible losses and loss functions.

This file is a copy of: https://github.com/mapbox/robosat/blob/master/robosat/losses.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()
class CrossEntropyLoss2d(nn.Module):
    """Cross-entropy.
    See: http://cs231n.github.io/neural-networks-2/#losses
    """

    def __init__(self, weight=None):
        """Creates an `CrossEntropyLoss2d` instance.
        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):
        return self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets)


class FocalLoss2d(nn.Module):
    """Focal Loss.
    Reduces loss for well-classified samples putting focus on hard mis-classified samples.
    See: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2, weight=None):
        """Creates a `FocalLoss2d` instance.
        Args:
          gamma: the focusing parameter; if zero this loss is equivalent with `CrossEntropyLoss2d`.
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)
        self.gamma = gamma

    def forward(self, inputs, targets):
        penalty = (1 - nn.functional.softmax(inputs, dim=1)) ** self.gamma
        return self.nll_loss(penalty * nn.functional.log_softmax(inputs, dim=1), targets)


class mIoULoss2d(nn.Module):
    """mIoU Loss.
    See:
      - http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
      - http://www.cs.toronto.edu/~wenjie/papers/iccv17/mattyus_etal_iccv17.pdf
    """

    def __init__(self, weight=None):
        """Creates a `mIoULoss2d` instance.
        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):

        N, C, H, W = inputs.size()

        softs = nn.functional.softmax(inputs, dim=1).permute(1, 0, 2, 3)
        masks = torch.zeros(N, C, H, W).to(targets.device).scatter_(1, targets.view(N, 1, H, W), 1).permute(1, 0, 2, 3)

        inters = softs * masks
        unions = (softs + masks) - (softs * masks)

        miou = 1. - (inters.view(C, N, -1).sum(2) / unions.view(C, N, -1).sum(2)).mean()

        return max(miou, self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets))


class LovaszLoss2d(nn.Module):
    """Lovasz Loss.
    See: https://arxiv.org/abs/1705.08790
    """

    def __init__(self):
        """Creates a `LovaszLoss2d` instance."""
        super().__init__()

    def forward(self, inputs, targets):

        N, C, H, W = inputs.size()
        masks = torch.zeros(N, C, H, W).to(targets.device).scatter_(1, targets.view(N, 1, H, W), 1)

        loss = 0.

        for mask, input in zip(masks.view(N, -1), inputs.view(N, -1)):

            max_margin_errors = 1. - ((mask * 2 - 1) * input)
            errors_sorted, indices = torch.sort(max_margin_errors, descending=True)
            labels_sorted = mask[indices.data]

            inter = labels_sorted.sum() - labels_sorted.cumsum(0)
            union = labels_sorted.sum() + (1. - labels_sorted).cumsum(0)
            iou = 1. - inter / union

            p = len(labels_sorted)
            if p > 1:
                iou[1:p] = iou[1:p] - iou[0:-1]

            loss += torch.dot(nn.functional.relu(errors_sorted), iou)

        return loss / N