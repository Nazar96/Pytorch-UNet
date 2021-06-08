from torch.nn import functional as F


def dice_loss(inputs, targets, smooth=1):
    # comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice


def dice_bce_loss(inputs, targets, smooth=1):
    # comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    Dice_BCE = BCE + dice_loss

    return Dice_BCE


supported_loss = {
    'bce': F.binary_cross_entropy,
    'mse': F.mse_loss,
    'dice_bce': dice_bce_loss,
}
