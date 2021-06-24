from torch.nn import functional as F
import torch


def dice_loss(inputs, targets, smooth=1):
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    result = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return result


def iou_loss(inputs, targets, smooth=1):
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    result = 1 - (intersection + smooth) / (union + smooth)
    return result


def dice_bce_loss(inputs, targets, smooth=1):
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    DICE = dice_loss(inputs, targets, smooth)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    result = BCE + DICE
    return result


def dice_mse_loss(inputs, targets, smooth=1):
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    DICE = dice_loss(inputs, targets, smooth)
    MSE = F.mse_loss(inputs, targets, reduction='mean')
    result = MSE + DICE
    return result


def iou_bce_loss(inputs, targets, smooth=1):
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    IOU = iou_loss(inputs, targets, smooth)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    result = IOU + BCE
    return result


def iou_mse_loss(inputs, targets, smooth=1):
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    IOU = iou_loss(inputs, targets, smooth)
    MSE = F.mse_loss(inputs, targets, reduction='mean')
    result = IOU + MSE
    return result


def focal_loss(inputs, targets, alpha=0.8, gamma=2, smooth=1):
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # first compute binary cross-entropy
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

    return focal_loss


def axis_std(tensor):
    std_x = tensor.mean(axis=2).std(axis=-1).mean()
    std_y = tensor.mean(axis=3).std(axis=-1).mean()
    res = (std_x + std_y)/2
    return res


supported_loss = {
    'bce': F.binary_cross_entropy,
    'mse': F.mse_loss,
    'dice_bce': dice_bce_loss,
    'dice_mse': dice_mse_loss,
    'dice': dice_loss,
    'iou': iou_loss,
    'iou_bce': iou_bce_loss,
    'iou_mse': iou_mse_loss,
    'focal': focal_loss,
}
