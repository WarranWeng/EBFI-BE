import numpy as np
import torch
from multiprocessing import Pool
# import MinkowskiEngine as ME
# from .cython_event_redistribute import event_redistribute as c_event_redistribute


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)


def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
        :param xs: tensor of x coords of events
        :param ys: tensor of y coords of events
        :param ps: tensor of event polarities/weights
        :param device: the device on which the image is. If none, set to events device
        :param sensor_size: the size of the image sensor/output image
        :param clip_out_of_range: if the events go beyond the desired image size,
            clip the events to fit into the image
        :param interpolation: which interpolation to use. Options=None,'bilinear'
        :param padding if bilinear interpolation, allow padding the image by 1 to allow events to fit:
    """
    # xs = xs - 1
    # ys = ys - 1
    
    xs_mask = (xs >= sensor_size[1]) + (xs < 0) 
    ys_mask = (ys >= sensor_size[0]) + (ys < 0) 
    mask = xs_mask + ys_mask
    xs[mask] = 0
    ys[mask] = 0
    ps[mask] = 0
    
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = torch.zeros(img_size).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        img.index_put_((ys, xs), ps, accumulate=True)
    return img


def binary_search_torch_tensor(t, l, r, x, side='left'):
    """
    Binary search sorted pytorch tensor
    """
    if r is None:
        r = len(t)-1
    while l <= r:
        if t[l] == x:
            return l
        if t[r] == x:
            return r
            
        mid = l + (r - l)//2;
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


def events_to_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    device : device to put voxel grid. If left empty, same device as events
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device

    if ts.sum() == 0 or len(ts) <= 3:
        return torch.zeros([B, sensor_size[0], sensor_size[1]], device=device)

    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0] + 1e-6
    delta_t = dt / B
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bi))
            weights = ps*bilinear_weights
            vb = events_to_image_torch(xs, ys,
                    weights, device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        else:
            tstart = ts[0] + delta_t*bi
            tend = tstart + delta_t
            beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart) 
            end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend, side='right') + 1
            vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                    ps[beg:end], device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins


def events_to_stack_polarity(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240)):
    """
    xs: torch.tensor, [xs]
    ys: torch.tensor, [ys]
    ts: torch.tensor, [ts]
    ps: torch.tensor, [ps]
    B: int, number of bins in output voxel grids 
    device: str or torch.device, device to put voxel grid. If left empty, same device as events
    sensor_size: tuple, the size of the event sensor/output voxels

    Returns: stack: torch.tensor, 2xBxHxW, stack of the events between t0 and t1, 2 refers to two polarities
    """
    if device is None:
        device = xs.device

    if ts.sum() == 0 or len(ts) <= 3:
        return torch.zeros([2, B, sensor_size[0], sensor_size[1]], device=device)

    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    positives = []
    negtives = []
    dt = ts[-1]-ts[0] + 1e-6
    delta_t = dt / B
    for bi in range(B):
        tstart = ts[0] + delta_t*bi
        tend = tstart + delta_t
        beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart) 
        end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend, side='right') + 1

        mask_pos = ps[beg:end].clone()
        mask_neg = ps[beg:end].clone()
        mask_pos[ps[beg:end] < 0] = 0
        mask_neg[ps[beg:end] > 0] = 0

        vp = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end] * mask_pos, device, sensor_size=sensor_size,
                clip_out_of_range=False)
        vn = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end] * mask_neg, device, sensor_size=sensor_size,
                clip_out_of_range=False)

        positives.append(vp)
        negtives.append(vn)

    positives_b = torch.stack(positives)
    negtives_b = torch.stack(negtives)
    stack = torch.stack([positives_b, negtives_b])

    return stack


def events_to_stack_no_polarity(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240)):
    """
    xs: torch.tensor, [xs]
    ys: torch.tensor, [ys]
    ts: torch.tensor, [ts]
    ps: torch.tensor, [ps]
    B: int, number of bins in output voxel grids 
    device: str or torch.device, device to put voxel grid. If left empty, same device as events
    sensor_size: tuple, the size of the event sensor/output voxels

    Returns: stack: torch.tensor, BxHxW, stack of the events between t0 and t1
    """
    if device is None:
        device = xs.device

    if ts.sum() == 0 or len(ts) <= 3:
        return torch.zeros([B, sensor_size[0], sensor_size[1]], device=device)

    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0] + 1e-6
    delta_t = dt / B
    for bi in range(B):
        tstart = ts[0] + delta_t*bi
        tend = tstart + delta_t
        beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart) 
        end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend, side='right') + 1

        b = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end], device, sensor_size=sensor_size,
                clip_out_of_range=False)

        bins.append(b)

    stack = torch.stack(bins)

    return stack

# **************************************
def events_to_image(xs, ys, ps, sensor_size=(180, 240)):
    """
    Accumulate events into an image.
    xs, ys, ps: torch.tensor, [N]
    """
    # xs = xs - 1
    # ys = ys - 1

    xs_mask = (xs >= sensor_size[1]) + (xs < 0) 
    ys_mask = (ys >= sensor_size[0]) + (ys < 0) 
    mask = xs_mask + ys_mask
    xs[mask] = 0
    ys[mask] = 0
    ps[mask] = 0

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=True)

    return img


def events_to_voxel(xs, ys, ts, ps, num_bins, sensor_size=(180, 240)):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """

    assert len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps)

    voxel = []
    ts = ts * (num_bins - 1)
    zeros = torch.zeros(ts.size())
    for b_idx in range(num_bins):
        weights = torch.max(zeros, 1.0 - torch.abs(ts - b_idx))
        voxel_bin = events_to_image(xs, ys, ps * weights, sensor_size=sensor_size)
        voxel.append(voxel_bin)

    return torch.stack(voxel)


def events_to_channels(xs, ys, ps, sensor_size=(180, 240)):
    """
    Generate a two-channel event image containing event counters.
    """

    assert len(xs) == len(ys) and len(ys) == len(ps)

    mask_pos = ps.clone()
    mask_neg = ps.clone()
    mask_pos[ps < 0] = 0
    mask_neg[ps > 0] = 0

    pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
    neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)

    return torch.stack([pos_cnt, neg_cnt])


def events_to_stack(xs, ys, ts, ps, B, sensor_size=(180, 240)):
    """
    xs: torch.tensor, [xs]
    ys: torch.tensor, [ys]
    ts: torch.tensor, [ts]
    ps: torch.tensor, [ps]
    B: int, number of bins in output voxel grids 
    device: str or torch.device, device to put voxel grid. If left empty, same device as events
    sensor_size: tuple, the size of the event sensor/output voxels

    Returns: stack: torch.tensor, 2xBxHxW, stack of the events between t0 and t1, 2 refers to two polarities
    """
    if ts.sum() == 0 or len(ts) <= 3:
        return torch.zeros([2, B, sensor_size[0], sensor_size[1]])

    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    positives = []
    negtives = []
    dt = ts[-1]-ts[0] + 1e-6
    delta_t = dt / B
    for bi in range(B):
        tstart = ts[0] + delta_t*bi
        tend = tstart + delta_t
        beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart) 
        end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend, side='right') + 1

        mask_pos = ps[beg:end].clone()
        mask_neg = ps[beg:end].clone()
        mask_pos[ps[beg:end] < 0] = 0
        mask_neg[ps[beg:end] > 0] = 0

        vp = events_to_image(xs[beg:end], ys[beg:end],
                ps[beg:end] * mask_pos, sensor_size=sensor_size)
        vn = events_to_image(xs[beg:end], ys[beg:end],
                ps[beg:end] * mask_neg, sensor_size=sensor_size)

        positives.append(vp)
        negtives.append(vn)

    positives_b = torch.stack(positives)
    negtives_b = torch.stack(negtives)
    stack = torch.stack([positives_b, negtives_b])

    return stack


def events_to_mask(xs, ys, ps, sensor_size=(180, 240)):
    """
    Accumulate events into a binary mask.
    """
    # xs = xs - 1
    # ys = ys - 1

    xs_mask = (xs >= sensor_size[1]) + (xs < 0) 
    ys_mask = (ys >= sensor_size[0]) + (ys < 0) 
    mask = xs_mask + ys_mask
    xs[mask] = 0
    ys[mask] = 0
    ps[mask] = 0

    device = xs.device
    img_size = list(sensor_size)
    mask = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    mask.index_put_((ys, xs), ps.abs(), accumulate=False)

    return mask


def events_polarity_mask(ps):
    """
    Creates a two channel tensor that acts as a mask for the input event list.
    :param ps: [N] tensor with event polarity ([-1, 1])
    :return [N x 2] event representation
    """
    inp_pol_mask = torch.stack([ps, ps])
    inp_pol_mask[0, :][inp_pol_mask[0, :] < 0] = 0
    inp_pol_mask[1, :][inp_pol_mask[1, :] > 0] = 0
    inp_pol_mask[1, :] *= -1

    return inp_pol_mask.transpose(0, 1)


def get_hot_event_mask(event_rate, idx, max_px=100, min_obvs=5, max_rate=0.8):
    """
    Returns binary mask to remove events from hot pixels.
    """

    mask = torch.ones(event_rate.shape).to(event_rate.device)
    if idx > min_obvs:
        for i in range(max_px):
            argmax = torch.argmax(event_rate)
            index = (argmax // event_rate.shape[1], argmax % event_rate.shape[1])
            if event_rate[index] > max_rate:
                event_rate[index] = 0
                mask[index] = 0
            else:
                break
    return mask


def stack2cnt(stack):
    """
    stack: torch.tensor, BxTBxHxW
    return: torch.tensor, Bx2xHxW, 0 for positive, 1 for negtive
    """
    stack = stack.clone().detach().round().cpu()

    pos = stack.clone()
    neg = stack.clone()
    pos[pos<0] = 0
    neg[neg>0] = 0
    neg *= -1

    pos = pos.sum(1)
    neg = neg.sum(1)

    cnt = torch.stack([pos, neg], dim=1)

    return cnt


if __name__ == '__main__':

    batch = 1
    bins = 10
    sensor_size = [4, 4]

    event_stack = torch.randint(-5, 15, [batch, bins]+sensor_size).float()
    print(event_stack)

    event_cloud = python_event_redistribute_NoPolarityStack(event_stack, mode='random')
    print(event_cloud)
    print(event_cloud.size())

    event_stack1 = events_to_voxel_torch(xs=event_cloud[0, 0, :, 0],
                                         ys=event_cloud[0, 0, :, 1],
                                         ts=event_cloud[0, 0, :, 2],
                                         ps=event_cloud[0, 0, :, 3],
                                         B=bins,
                                         sensor_size=sensor_size,
                                         temporal_bilinear=False)

    print(event_stack1)

    print((event_stack-event_stack1).sum())