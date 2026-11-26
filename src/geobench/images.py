"""Shared image decoding, identity and classifier/diffusion geometry."""
import hashlib
import math
import struct

import numpy as np
import PIL
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def decode(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def decoded_pixel_sha256(path):
    image = decode(path)
    return hashlib.sha256(b'RGB8\0' + struct.pack('>QQ', *image.size) + image.tobytes()).hexdigest()


def _tensor(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(ImageOps.exif_transpose(image).convert('RGB'), copy=True)).permute(2, 0, 1).float().div(255)
    return torch.as_tensor(image)


def classifier_geometry(image):
    x = _tensor(image)
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError('expected RGB CHW image')
    _, h, w = x.shape
    scale = min(896 / w, 384 / h)
    nw = min(896, max(1, math.floor(w * scale + .5)))
    nh = min(384, max(1, math.floor(h * scale + .5)))
    resized = resize(x, [nh, nw], interpolation=InterpolationMode.BILINEAR, antialias=True)
    canvas = torch.tensor(MEAN, dtype=x.dtype, device=x.device)[:, None, None].expand(3, 384, 896).clone()
    left, top = (896 - nw) // 2, (384 - nh) // 2
    canvas[:, top:top + nh, left:left + nw] = resized
    return canvas


def normalize_classifier(x):
    mean = x.new_tensor(MEAN).view(3, 1, 1)
    std = x.new_tensor(STD).view(3, 1, 1)
    return (x - mean) / std


def pad_native(image):
    x = _tensor(image)
    _, h, w = x.shape
    dw, dh = (-w) % 8, (-h) % 8
    left, top = dw // 2, dh // 2
    right, bottom = dw - left, dh - top
    if max(left, right) >= w or max(top, bottom) >= h:
        raise ValueError('reflection padding exceeds image size')
    result = F.pad(x, (left, right, top, bottom), mode='reflect')
    return result, {'original_size': [w, h], 'padded_size': [w + dw, h + dh],
                    'offsets': [left, top, right, bottom], 'crop_bbox': [left, top, left + w, top + h]}


def crop_native(x, padding):
    left, top, right, bottom = padding['crop_bbox']
    if list(x.shape[-2:]) != list(reversed(padding['padded_size'])):
        raise ValueError('padded size mismatch')
    return x[..., top:bottom, left:right]
