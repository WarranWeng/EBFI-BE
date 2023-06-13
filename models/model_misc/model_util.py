"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import os
import copy
from math import fabs, ceil, floor

import numpy as np
import torch
from torch.nn import ZeroPad2d, ConstantPad3d
import torch.nn.init as init
import torch.nn as nn


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.GroupNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def skip_concat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    padding = ZeroPad2d((diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
    x1 = padding(x1)
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    padding = ZeroPad2d((diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
    x1 = padding(x1)
    return x1 + x2


def optimal_crop_size(max_size, max_subsample_factor, safety_margin=0):
    """
    Find the optimal crop size for a given max_size and subsample_factor.
    The optimal crop size is the smallest integer which is greater or equal than max_size,
    while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    crop_size += safety_margin * pow(2, max_subsample_factor)
    return crop_size


def OptimalCropSize(max_size, factor, safety_margin=0):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by factor.
    """
    crop_size = int(factor * ceil(max_size / factor))
    crop_size += safety_margin * factor
    return crop_size


class CropParameters:
    """
    Helper class to compute and store useful parameters for pre-processing and post-processing
    of images in and out of E2VID.
    Pre-processing: finding the best image size for the network, and padding the input image with zeros
    Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders, safety_margin=0):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders, safety_margin)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders, safety_margin)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ZeroPad2d(
            (
                self.padding_left,
                self.padding_right,
                self.padding_top,
                self.padding_bottom,
            )
        )

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)

    def crop(self, img):
        return img[..., self.iy0 : self.iy1, self.ix0 : self.ix1]


class ScaleCropParameters:
    """
    Helper class to compute and store useful parameters for pre-processing and post-processing
    of images in and out of E2VID.
    Pre-processing: finding the best image size for the network, and padding the input image with zeros
    Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders, scale, safety_margin=0):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders, safety_margin)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders, safety_margin)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ZeroPad2d(
            (
                self.padding_left,
                self.padding_right,
                self.padding_top,
                self.padding_bottom,
            )
        )

        self.cx = floor(self.width_crop_size * scale / 2) 
        self.cy = floor(self.height_crop_size * scale / 2) 

        self.ix0 = self.cx - floor(self.width * scale / 2)
        self.ix1 = self.cx + ceil(self.width * scale / 2)
        self.iy0 = self.cy - floor(self.height * scale / 2)
        self.iy1 = self.cy + ceil(self.height * scale / 2)

    def crop(self, img):
        return img[..., self.iy0 : self.iy1, self.ix0 : self.ix1]


class CropSize:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, patch_size, scale=1, safety_margin=0):

        self.scale = scale
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.width_crop_size = OptimalCropSize(self.width, patch_size['w'], safety_margin)
        self.height_crop_size = OptimalCropSize(self.height, patch_size['h'], safety_margin)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ZeroPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

    def crop(self, img):
        cx = floor(self.width_crop_size * self.scale / 2)
        cy = floor(self.height_crop_size * self.scale / 2)

        ix0 = cx - floor(self.width * self.scale / 2)
        ix1 = cx + ceil(self.width * self.scale / 2)
        iy0 = cy - floor(self.height * self.scale / 2)
        iy1 = cy + ceil(self.height * self.scale / 2)

        return img[..., iy0:iy1, ix0:ix1]


class CropSize3D:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, depth, width, height, patch_size, scale=1, safety_margin=0):

        self.scale = scale
        self.depth = depth
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.depth_crop_size = OptimalCropSize(self.depth, patch_size['d'], safety_margin)
        self.width_crop_size = OptimalCropSize(self.width, patch_size['w'], safety_margin)
        self.height_crop_size = OptimalCropSize(self.height, patch_size['h'], safety_margin)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.padding_front = ceil(0.5 * (self.depth_crop_size - self.depth))
        self.padding_back = floor(0.5 * (self.depth_crop_size - self.depth))
        self.pad = ConstantPad3d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom, self.padding_front, self.padding_back), 0)

    def crop(self, img):
        cd = floor(self.depth_crop_size * self.scale / 2)
        cx = floor(self.width_crop_size * self.scale / 2)
        cy = floor(self.height_crop_size * self.scale / 2)

        id0 = cd - floor(self.depth * self.scale / 2)
        id1 = cd + ceil(self.depth * self.scale / 2)
        ix0 = cx - floor(self.width * self.scale / 2)
        ix1 = cx + ceil(self.width * self.scale / 2)
        iy0 = cy - floor(self.height * self.scale / 2)
        iy1 = cy + ceil(self.height * self.scale / 2)

        return img[..., id0:id1, iy0:iy1, ix0:ix1]


def recursive_clone(tensor):
    """
    Assumes tensor is a torch.tensor with 'clone()' method, possibly
    inside nested iterable.
    E.g., tensor = [(pytorch_tensor, pytorch_tensor), ...]
    """
    if hasattr(tensor, "clone"):
        return tensor.clone()
    try:
        return type(tensor)(recursive_clone(t) for t in tensor)
    except TypeError:
        print("{} is not iterable and has no clone() method.".format(tensor))


def copy_states(states):
    """
    LSTM states: [(torch.tensor, torch.tensor), ...]
    GRU states: [torch.tensor, ...]
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)
