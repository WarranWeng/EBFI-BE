import torch
import torch.nn.functional as F
import torchvision.transforms
from math import sin, cos, pi
import numbers
import numpy as np
import random
from typing import Union


# class RandomCrop(object):
#     """Crop the tensor at a random location.
#     """

#     def __init__(self, size, preserve_mosaicing_pattern=False):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size

#         self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

#     @staticmethod
#     def get_params(x, output_size):
#         w, h = x.shape[2], x.shape[1]
#         th, tw = output_size
#         assert th % 4 == 0 and tw % 4 == 0
#         if th > h or tw > w:
#             raise Exception("Input size {}x{} is less than desired cropped \
#                     size {}x{} - input tensor shape = {}".format(w,h,tw,th,x.shape))
#         if w == tw and h == th:
#             return 0, 0, h, w

#         i = random.randint(0, h - th)
#         j = random.randint(0, w - tw)

#         i = int(i // 4) * 4
#         j = int(j // 4) * 4

#         return i, j, th, tw

#     def crop(self, x, i, j, h, w):
#         """
#             x: [C x H x W] Tensor to be rotated.
#         Returns:
#             Tensor: Cropped tensor.
#         """
#         if self.preserve_mosaicing_pattern:
#             # make sure that i and j are even, to preserve the mosaicing pattern
#             if i % 2 == 1:
#                 i = i + 1
#             if j % 2 == 1:
#                 j = j + 1

#         return x[:, i:i + h, j:j + w]

#     def __call__(self, pred_frame, ori_left_frame, down2_left_frame, down4_left_frame,
#                         ori_right_frame, down2_right_frame, down4_right_frame,
#                         hr_event_cnt, ori_lr_event_cnt, down2_lr_event_cnt, down4_lr_event_cnt):
#         assert pred_frame.size()[-2:] == hr_event_cnt.size()[-2:]

#         i, j, h, w = self.get_params(pred_frame, self.size)

#         pred_frame = self.crop(pred_frame, i, j, h, w)
#         ori_left_frame = self.crop(ori_left_frame, i, j, h, w)
#         down2_left_frame = self.crop(down2_left_frame, i//2, j//2, h//2, w//2)
#         down4_left_frame = self.crop(down4_left_frame, i//4, j//4, h//4, w//4)
#         ori_right_frame = self.crop(ori_right_frame, i, j, h, w)
#         down2_right_frame = self.crop(down2_right_frame, i//2, j//2, h//2, w//2)
#         down4_right_frame = self.crop(down4_right_frame, i//4, j//4, h//4, w//4)

#         hr_event_cnt = self.crop(hr_event_cnt, i, j, h, w)
#         ori_lr_event_cnt = self.crop(ori_lr_event_cnt, i, j, h, w)
#         down2_lr_event_cnt = self.crop(down2_lr_event_cnt, i//2, j//2, h//2, w//2)
#         down4_lr_event_cnt = self.crop(down4_lr_event_cnt, i//4, j//4, h//4, w//4)

#         return pred_frame, ori_left_frame, down2_left_frame, down4_left_frame, \
#                         ori_right_frame, down2_right_frame, down4_right_frame, \
#                         hr_event_cnt, ori_lr_event_cnt, down2_lr_event_cnt, down4_lr_event_cnt

#     def __repr__(self):
#         return self.__class__.__name__ + '(size={0})'.format(self.size)


def RandomCrop(size, scale,
                pred_frame, 
                ori_left_frame, down2_left_frame, down4_left_frame, 
                ori_right_frame, down2_right_frame, down4_right_frame, 
                hr_event_cnt, lr_event_cnt, lr_scaled_event_cnt, 
                ori_left_lr_event_cnt, down2_left_lr_event_cnt, down4_left_lr_event_cnt, 
                ori_right_lr_event_cnt, down2_right_lr_event_cnt, down4_right_lr_event_cnt
                ):
    def get_params(x, output_size):
        w, h = x.shape[2], x.shape[1]
        th, tw = output_size
        assert th % 4 == 0 and tw % 4 == 0
        if th > h or tw > w:
            raise Exception("Input size {}x{} is less than desired cropped \
                    size {}x{} - input tensor shape = {}".format(w,h,tw,th,x.shape))
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        i = int(i // 4) * 4
        j = int(j // 4) * 4

        return i, j, th, tw

    def crop(x, i, j, h, w):
        """
            x: [C x H x W] Tensor to be rotated.
        Returns:
            Tensor: Cropped tensor.
        """

        return x[:, i:i + h, j:j + w]

    assert pred_frame.size()[-2:] == hr_event_cnt.size()[-2:]

    i, j, h, w = get_params(pred_frame, size)

    pred_frame = crop(pred_frame, i, j, h, w)
    ori_left_frame = crop(ori_left_frame, i, j, h, w)
    down2_left_frame = crop(down2_left_frame, i//2, j//2, h//2, w//2)
    down4_left_frame = crop(down4_left_frame, i//4, j//4, h//4, w//4)
    ori_right_frame = crop(ori_right_frame, i, j, h, w)
    down2_right_frame = crop(down2_right_frame, i//2, j//2, h//2, w//2)
    down4_right_frame = crop(down4_right_frame, i//4, j//4, h//4, w//4)

    hr_event_cnt = crop(hr_event_cnt, i, j, h, w)
    lr_event_cnt = crop(lr_event_cnt, i//scale, j//scale, h//scale, w//scale)
    lr_scaled_event_cnt = crop(lr_scaled_event_cnt, i, j, h, w)
    ori_left_lr_event_cnt = crop(ori_left_lr_event_cnt, i, j, h, w)
    down2_left_lr_event_cnt = crop(down2_left_lr_event_cnt, i//2, j//2, h//2, w//2)
    down4_left_lr_event_cnt = crop(down4_left_lr_event_cnt, i//4, j//4, h//4, w//4)
    ori_right_lr_event_cnt = crop(ori_right_lr_event_cnt, i, j, h, w)
    down2_right_lr_event_cnt = crop(down2_right_lr_event_cnt, i//2, j//2, h//2, w//2)
    down4_right_lr_event_cnt = crop(down4_right_lr_event_cnt, i//4, j//4, h//4, w//4)

    return pred_frame, \
            ori_left_frame, down2_left_frame, down4_left_frame, \
            ori_right_frame, down2_right_frame, down4_right_frame, \
            hr_event_cnt, lr_event_cnt, lr_scaled_event_cnt, \
            ori_left_lr_event_cnt, down2_left_lr_event_cnt, down4_left_lr_event_cnt, \
            ori_right_lr_event_cnt, down2_right_lr_event_cnt, down4_right_lr_event_cnt
