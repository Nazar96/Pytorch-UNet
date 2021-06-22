from os.path import splitext
from os import listdir
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2
from random import choice
import numpy as np
import os


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')


class TableMaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir, background_dir=None, labels=None, transform=None, background_transform=None,
                 back_labels=None, back_proba=0.5, empty_proba=0.0):
        self.img_labels = labels
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.background_dir = background_dir
        self.background_transform = background_transform
        self.back_labels = back_labels
        self.sigma = 3
        self.empty_proba = empty_proba
        self.back_proba = back_proba

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        mask_path = os.path.join(self.mask_dir, self.img_labels[idx])

        is_empty = np.random.binomial(1, self.empty_proba)
        is_background = np.random.binomial(1, self.back_proba)

        if is_empty:
            img_path = mask_path

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        mask = cv2.GaussianBlur(mask, (15, 15), self.sigma)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if (self.background_transform is not None) and (is_background):
            image = self._add_background(image)

        image = self._postproc(image)
        mask = 1 - self._postproc(mask)

        return image, mask

    @staticmethod
    def _postproc(x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = cv2.normalize(x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        x = np.expand_dims(x, axis=0)
        return x

    def _add_background(self, img):
        back_rand = choice(self.back_labels)
        back_path = os.path.join(self.background_dir, back_rand)
        back = cv2.imread(back_path)
        back = self.background_transform(image=back)['image']
        result = np.mean([back, img], axis=0).astype(np.uint8)
        return result

