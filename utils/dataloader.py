import os
import albumentations as A
import cv2
from sklearn.model_selection import train_test_split
from .dataset import TableMaskDataset
from torch.utils.data import DataLoader

img_pad_size = 1024
img_size = 512
val_size = 50
batch_size = 16

pixel_aug = [
    A.Perspective(p=0.25),
    # A.GlassBlur(p=0.1),
    A.RandomGridShuffle(p=0.15),
    A.RandomRain(rain_type='heavy', p=0.15),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, p=0.15),
    A.RandomSunFlare(src_radius=200, num_flare_circles_lower=1, p=0.25),
    A.OpticalDistortion(distort_limit=0.15, shift_limit=0, p=0.25),
]

pad_aug = [
    A.LongestMaxSize(img_pad_size, p=0.25),

    A.OneOf([
        A.PadIfNeeded(
            min_height=img_pad_size,
            min_width=img_pad_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
            mask_value=(255, 255, 255),
        ),
        A.PadIfNeeded(
            min_height=img_pad_size,
            min_width=img_pad_size,
            border_mode=cv2.BORDER_REFLECT,
        )], p=1,
    ),
    A.RandomCrop(width=img_size, height=img_size, p=1),
]

back_aug = [
    A.ShiftScaleRotate(0, 1, rotate_limit=180, p=0.75),
    A.PadIfNeeded(
        min_height=img_pad_size,
        min_width=img_pad_size,
        border_mode=cv2.BORDER_REFLECT,
    ),
    A.RandomCrop(width=img_size, height=img_size, p=1),
]

affin_aug = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(0.5, 0.5, rotate_limit=0.2, p=0.5),
    A.RandomRotate90(p=0.5),
]

aug_list = pixel_aug + pad_aug + affin_aug
aug_back_list = pixel_aug + back_aug

train_alb_transform = A.Compose(aug_list)
val_alb_transform = A.Compose(pad_aug)
back_alb_transform = A.Compose(aug_back_list)


def get_dataloader(
        img_path='./test/type_1/',
        mask_path='./test/type_2/',
        background_path='./background/',
):
    labels = os.listdir(img_path)
    backgrounds = os.listdir(background_path)

    train_labels, val_labels = train_test_split(labels, test_size=val_size, random_state=228)

    train_ds = TableMaskDataset(
        img_path, mask_path, background_path, train_labels,
        transform=train_alb_transform, background_transform=back_alb_transform, back_labels=backgrounds,
        back_proba=0.5,
    )
    val_ds = TableMaskDataset(
        img_path, mask_path, background_path, val_labels,
        transform=val_alb_transform, background_transform=back_alb_transform, back_labels=backgrounds,
        back_proba=0.5,
    )

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=val_size, shuffle=False)

    return train_dataloader, val_dataloader
