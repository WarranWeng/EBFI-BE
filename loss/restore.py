import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import torch.nn.functional as F
# local modules
from .PerceptualSimilarity import models


class perceptual_loss():
    def __init__(self, weight=1.0, net='alex', use_gpu=True, gpu_ids=[0]):
        """
        Wrapper for PerceptualSimilarity.models.PerceptualLoss
        """
        self.model = models.PerceptualLoss(net=net, use_gpu=use_gpu, gpu_ids=gpu_ids)
        self.weight = weight

    def __call__(self, pred, target, normalize=True):
        """
        pred and target are Tensors with shape N x C x H x W (C {1, 3})
        normalize scales images from [0, 1] to [-1, 1] (default: True)
        PerceptualLoss expects N x 3 x H x W.
        """
        assert pred.size() == target.size()

        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
            target = torch.cat([target, target, target], dim=1)
            dist = self.model.forward(pred, target, normalize=normalize)
        elif pred.shape[1] == 3:
            dist = self.model.forward(pred, target, normalize=normalize)
        else:
            num_ch = pred.shape[1]
            dist = 0
            for idx in range(num_ch):
                dist += self.model.forward(pred[:, idx].repeat(1, 3, 1, 1), target[:, idx].repeat(1, 3, 1, 1), normalize=normalize)
            dist /= num_ch

        return self.weight * dist.mean()


class ssim_loss():
    def __init__(self):
        self.ssim = SSIM

    def __call__(self, pred, tgt):
        """
        pred, tgt: torch.tensor, 1xNxHxW
        """
        assert pred.size() == tgt.size()
        pred = pred.squeeze().cpu().numpy()
        tgt = tgt.squeeze().cpu().numpy()

        if len(pred.shape) == 3:
            num_ch = pred.shape[0]
            loss = 0
            for idx in range(num_ch):
                loss += self.ssim(pred[idx], tgt[idx])
            loss /= num_ch
        else:
            loss = self.ssim(pred, tgt)

        return loss


class psnr_loss():
    def __init__(self):
        self.psnr = PSNR

    def __call__(self, pred, tgt):
        """
        pred, tgt: torch.tensor, 1xNxHxW
        """
        assert pred.size() == tgt.size()
        pred = pred.squeeze().cpu().numpy()
        tgt = tgt.squeeze().cpu().numpy()

        if len(pred.shape) == 3:
            num_ch = pred.shape[0]
            loss = 0
            for idx in range(num_ch):
                # data_range = max(tgt[idx].max()-tgt.min(), pred[idx].max()-pred[idx].min())
                data_range = tgt[idx].max()-tgt.min()
                loss += self.psnr(tgt[idx], pred[idx], data_range=data_range)
            loss /= num_ch
        else:
            loss = self.psnr(pred.clip(0, 1), tgt.clip(0, 1))

        # loss = self.psnr((tgt.squeeze().cpu().numpy()*255).astype(np.uint8), (pred.squeeze().cpu().numpy()*255).astype(np.uint8), data_range=255)

        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


class Ternary(nn.Module):
    def __init__(self, patch_size=7):
        super(Ternary, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        if torch.cuda.is_available():
            self.w = torch.tensor(self.w).float().cuda()

    def transform(self, tensor):
        tensor_ = tensor.mean(dim=1, keepdim=True)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size//2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_norm = loc_diff / torch.sqrt(0.81 + loc_diff ** 2)

        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size//2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        
        return mask
        
    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y.detach()
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()

        return loss


# laplacian loss
class GaussianConv(nn.Module):
    def __init__(self):
        super(GaussianConv, self).__init__()
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        self.kernel = nn.Parameter(kernel.div(256).repeat(3,1,1,1), requires_grad=False)

    def forward(self, x, factor=1):
        c, h, w = x.shape[1:]
        p = (self.kernel.shape[-1]-1)//2
        blurred = F.conv2d(F.pad(x, pad=(p,p,p,p), mode='reflect'), factor*self.kernel, groups=c)
        return blurred

class LaplacianPyramid(nn.Module):
    """
    Implementing "The Laplacian pyramid as a compact image code." Burt, Peter J., and Edward H. Adelson. 
    """
    def __init__(self, max_level=5):
        super(LaplacianPyramid, self).__init__()
        self.gaussian_conv = GaussianConv()
        self.max_level = max_level

    def forward(self, X):
        pyramid = []
        current = X
        for _ in range(self.max_level-1):
            blurred = self.gaussian_conv(current)
            reduced = self.reduce(blurred)
            expanded = self.expand(reduced)
            diff = current - expanded
            pyramid.append(diff)
            current = reduced

        pyramid.append(current)

        return pyramid
    
    def reduce(self, x):
        return F.avg_pool2d(x, 2)
    
    def expand(self, x):
        # injecting even zero rows
        tmp = torch.cat([x, torch.zeros_like(x).to(x.device)], dim=3)
        tmp = tmp.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
        tmp = tmp.permute(0,1,3,2)
        # injecting even zero columns
        tmp = torch.cat([tmp, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
        tmp = tmp.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
        x_up = tmp.permute(0,1,3,2)
        # convolve with 4 x Gaussian kernel
        return self.gaussian_conv(x_up, factor=4)

class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()

        self.criterion = nn.L1Loss(reduction='sum')
        self.lap = LaplacianPyramid()

    def forward(self, x, y):
        x_lap, y_lap = self.lap(x), self.lap(y)
        return sum(2**i * self.criterion(a, b) for i, (a, b) in enumerate(zip(x_lap, y_lap)))
