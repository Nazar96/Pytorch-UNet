from torch.nn import functional as F


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


supported_loss = {
    'bce': F.binary_cross_entropy,
    'mse': F.mse_loss,
    'dice_bce': dice_bce_loss,
    'dice_mse': dice_mse_loss,
    'dice': dice_loss,
    'iou': iou_loss,
    'iou_bce': iou_bce_loss,
}
