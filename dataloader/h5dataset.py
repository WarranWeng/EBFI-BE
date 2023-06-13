import torch
from torch.utils.data import Dataset
import torch.nn.functional as func
import os
from glob import glob
import h5py
import cv2
from tqdm import tqdm
import numpy as np
import random
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.style.use('seaborn-whitegrid')
# local modules
from dataloader.encodings import events_to_stack


class H5Dataset(Dataset):
    def __init__(self, h5_file_path, config):
        super().__init__()

        self.config = config
        self.h5_file_path = h5_file_path
        self.set_data_scale()
        self.load_metadata()
        self.set_period_items()
        self.set_items()

    def set_data_scale(self):
        self.h5_file = h5py.File(self.h5_file_path, 'r')
        self.sensor_resolution = self.h5_file.attrs['sensor_resolution'].tolist()
        self.scale = self.config['scale']
        self.ori_scale = self.config['ori_scale']

        self.gt_sensor_resolution = None
        self.gt_prex = None
        if self.ori_scale == 'ori':
            self.inp_sensor_resolution = self.sensor_resolution
            self.inp_prex = self.ori_scale
            if self.scale == 1:
                self.gt_sensor_resolution = self.sensor_resolution
                self.gt_prex = 'ori'
            else:
                raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

        elif self.ori_scale == 'down2':
            self.inp_sensor_resolution = [round(i / 2) for i in self.sensor_resolution]
            self.inp_prex = self.ori_scale
            if self.scale == 2:
                self.gt_sensor_resolution = self.sensor_resolution
                self.gt_prex = 'ori'
            else:
                raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

        elif self.ori_scale == 'down4':
            self.inp_sensor_resolution = [round(i / 4) for i in self.sensor_resolution]
            self.inp_prex = self.ori_scale
            if self.scale == 2:
                self.gt_sensor_resolution = [round(i / 2) for i in self.sensor_resolution]
                self.gt_prex = 'down2'
            elif self.scale == 4:
                self.gt_sensor_resolution = self.sensor_resolution
                self.gt_prex ='ori'
            else:
                raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

        elif self.ori_scale == 'down8':
            self.inp_sensor_resolution = [round(i / 8) for i in self.sensor_resolution]
            self.inp_prex = self.ori_scale
            if self.scale == 2:
                self.gt_sensor_resolution = [round(i / 4) for i in self.sensor_resolution]
                self.gt_prex = 'down4'
            elif self.scale == 4:
                self.gt_sensor_resolution = [round(i / 2) for i in self.sensor_resolution]
                self.gt_prex ='down2'
            elif self.scale == 8:
                self.gt_sensor_resolution = self.sensor_resolution
                self.gt_prex ='ori'
            else:
                raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

        elif self.ori_scale == 'down16':
            self.inp_sensor_resolution = [round(i / 16) for i in self.sensor_resolution]
            self.inp_prex = self.ori_scale
            if self.scale == 2:
                self.gt_sensor_resolution = [round(i / 8) for i in self.sensor_resolution]
                self.gt_prex = 'down8'
            elif self.scale == 4:
                self.gt_sensor_resolution = [round(i / 4) for i in self.sensor_resolution]
                self.gt_prex ='down4'
            elif self.scale == 8:
                self.gt_sensor_resolution = [round(i / 2) for i in self.sensor_resolution]
                self.gt_prex ='down2'
            elif self.scale == 16:
                self.gt_sensor_resolution = self.sensor_resolution
                self.gt_prex ='ori'
            else:
                raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

        else:
            raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

    def load_metadata(self):
        self.NumFramePerPeriod = self.config['NumFramePerPeriod']
        self.NumFramePerBlurry = self.config['NumFramePerBlurry']
        self.NumPeriodPerSeq = self.config['NumPeriodPerSeq']
        self.SlidingWindowSeq = self.config['SlidingWindowSeq']
        self.NumPeriodPerLoad = self.config['NumPeriodPerLoad']
        self.SlidingWindowLoad = self.config['SlidingWindowLoad']
        self.ExposureMethod = self.config['ExposureMethod']
        self.ExposureTime = self.config['ExposureTime'] # e.g., [3, 5, 7, 9]
        self.Interval = self.NumFramePerPeriod * self.NumPeriodPerLoad
        self.DeblurPretrain = self.config.get('DeblurPretrain', False)
        self.NeedNeighborGT = self.config.get('NeedNeighborGT', False)

        self.time_bins = self.config['time_bins']
        # self.skip_frame = self.config['skip_frame']
        self.num_imgs = len(self.h5_file['ori_images'].keys())

    def set_period_items(self):
        assert self.NumFramePerBlurry >= 1, 'Number of frames per blurry must >= 1!'
        assert self.NumFramePerPeriod >= 1, 'Number of frames per period must >= 1!'
        assert self.NumFramePerBlurry <= self.NumFramePerPeriod, 'Number of frames per blurry must <= Number of frames per period!'
        assert self.ExposureMethod in ['Fixed', 'Auto', 'Custom'], 'Error exposure setting!'
        self.PeriodIndices = []
        self.BlurryIndices = []
        self.LatentIndices = []
        self.NeighborIndices = []
        self.ExposureDuty = []
        candidates_indices = np.arange(0, self.num_imgs, self.NumFramePerPeriod)
        Length = len(candidates_indices) - 1
        if self.ExposureMethod == 'Custom':
            ExLength = len(self.ExposureTime)
        for j, idx in enumerate(candidates_indices[:-1]):
            self.PeriodIndices.append([idx, idx+self.NumFramePerPeriod-1])
            self.LatentIndices.append([idx+i for i in range(self.NumFramePerPeriod)])
            if self.NeedNeighborGT:
                TempList = []
                for i in range(self.NumFramePerPeriod):
                    if i == 0:
                        TempList.append([idx + i, idx + i + 1])
                    elif i == self.NumFramePerPeriod - 1:
                        TempList.append([idx + i - 1, idx + i])
                    else:
                        TempList.append([idx + i - 1, idx + i + 1])
                self.NeighborIndices.append(TempList)
            if self.ExposureMethod == 'Fixed':
                # self.BlurryIndices.append([idx, idx+self.NumFramePerBlurry-1])
                self.BlurryIndices.append([idx+i for i in range(self.NumFramePerBlurry)])
                self.ExposureDuty.append(torch.tensor(self.NumFramePerBlurry / self.NumFramePerPeriod))
            elif self.ExposureMethod == 'Auto':
                TempExp = np.random.randint(1, self.NumFramePerPeriod)
                # self.BlurryIndices.append([idx, idx+TempExp])
                self.BlurryIndices.append([idx+i for i in range(TempExp)])
                self.ExposureDuty.append(torch.tensor(TempExp / self.NumFramePerPeriod))
            elif self.ExposureMethod == 'Custom':
                Index = j % ExLength
                TempExp = self.ExposureTime[Index]
                assert TempExp <= self.NumFramePerPeriod, 'Number of frames per blurry must <= Number of frames per period!'
                self.BlurryIndices.append([idx+i for i in range(TempExp)])
                self.ExposureDuty.append(torch.tensor(TempExp / self.NumFramePerPeriod))
            else:
                raise Exception('Error exposure method!')

    def set_items(self):
        assert self.NumPeriodPerSeq >= 1, 'Number of period per seq must >= 1!'
        assert self.SlidingWindowSeq >= 0, 'Sliding window seq must >= 0!'
        assert self.SlidingWindowSeq <= self.NumPeriodPerSeq, 'Sliding window seq must <= number of period per seq'
        assert self.NumPeriodPerLoad >= 1, 'Number of period per Load must >= 1!'
        assert self.SlidingWindowLoad >= 0, 'Sliding window Load must >= 0!'
        assert self.SlidingWindowLoad <= self.NumPeriodPerLoad, 'Sliding window Load must <= number of period per Load'
        assert self.NumPeriodPerLoad <= self.NumPeriodPerSeq, 'Number of period per load must <= Number of period per seq'
        NumPeriod = len(self.PeriodIndices)
        self.SeqIndices = []
        LoadIndices = []
        candidates_indices = np.arange(0, NumPeriod, self.SlidingWindowSeq)
        for idx in candidates_indices:
            start, end = idx, idx + self.NumPeriodPerSeq -1
            if end <= NumPeriod -1:
                tmp_indices = np.arange(start, end+1, self.SlidingWindowLoad)
                for i in tmp_indices:
                    if i+self.NumPeriodPerLoad-1 <= end:
                        LoadIndices.append([i, i+self.NumPeriodPerLoad-1])
                self.SeqIndices.append(LoadIndices)
                LoadIndices = []

    def __len__(self):

        return len(self.SeqIndices)

    def __getitem__(self, index, seed=None):
        if seed is None:
            seed = random.randint(0, 2**32)

        Sequence = self.SeqIndices[index]
        SeqLatentFList = []
        SeqBlurryFList = []
        SeqNeighborFList = []
        # SeqLREvList = []
        # SeqLREvBiList = []
        SeqHREvList = []
        LatentTimestampList = []
        RelativeLatentTimestampList = []
        BlurryTimestampList = []
        SeqExposureDutyList = []
        for LoadIndex in Sequence:
            left, right = LoadIndex
            LatentFramesList = []
            BlurryFramesList = []
            NeighborFramesList = []
            AllLatentIndices = []
            AllBlurryIndices = []
            ExposureDutyList = []
            for i in range(left, right+1):
                LatentIndices = self.LatentIndices[i]
                AllLatentIndices += LatentIndices
                BlurryIndices = self.BlurryIndices[i]
                if self.NeedNeighborGT:
                    NeighborIndices = self.NeighborIndices[i]
                AllBlurryIndices.append(BlurryIndices)
                LatentFramesList.append(self.GetFrames(LatentIndices, 'sharp'))
                BlurryFramesList.append(self.GetFrames(BlurryIndices, 'blurry'))
                if self.NeedNeighborGT:
                    NeighborFramesList.append(self.GetFrames(NeighborIndices, 'neighbor'))
                ExposureDutyList.append(self.ExposureDuty[i])
            LatentFrames = torch.stack(LatentFramesList, dim=0) # NumPxNumFx3xHxW 
            BlurryFrames = torch.stack(BlurryFramesList, dim=0) # NumPx3xHxW
            if self.NeedNeighborGT:
                NeighborFrames = torch.stack(NeighborFramesList, dim=0) # NumPxNumFx2x3xHxW
            # LREvents, LREventsBi, HREvents = self.GetEvents(AllLatentIndices) # TBx2xHxW
            HREvents = self.GetEvents(AllLatentIndices) # TBx2xHxW
            LatentTimestamp, RelativeLatentTimestamp, BlurryTimestamp = self.GetTimestamp(AllLatentIndices, AllBlurryIndices)
            ExposureDuty = torch.stack(ExposureDutyList).unsqueeze(-1) # NumPx1
            SeqLatentFList.append(LatentFrames)
            SeqBlurryFList.append(BlurryFrames)
            if self.NeedNeighborGT:
                SeqNeighborFList.append(NeighborFrames)
            # SeqLREvList.append(LREvents)
            # SeqLREvBiList.append(LREventsBi)
            SeqHREvList.append(HREvents)
            LatentTimestampList.append(LatentTimestamp)
            RelativeLatentTimestampList.append(RelativeLatentTimestamp)
            BlurryTimestampList.append(BlurryTimestamp)
            SeqExposureDutyList.append(ExposureDuty)
        SeqLatentF = torch.stack(SeqLatentFList, dim=0) # LxNumPxNumFx3xHxW
        SeqBlurryF = torch.stack(SeqBlurryFList, dim=0) # LxNumPx3xHxW
        if self.NeedNeighborGT:
            SeqNeighborF = torch.stack(SeqNeighborFList, dim=0) # LxNumPxNumFx2x3xHxW
        # SeqLREv = torch.stack(SeqLREvList, dim=0) # LxTBx2xH1xW1
        # SeqLREvBi = torch.stack(SeqLREvBiList, dim=0) # LxTBx2xHxW
        SeqHREv = torch.stack(SeqHREvList, dim=0) # LxTBx2xHxW
        LatentTs = torch.stack(LatentTimestampList, dim=0) # LxNumF
        RelativeLatentTs = torch.stack(RelativeLatentTimestampList, dim=0) # LxNumPxNumF
        BlurryTs = torch.stack(BlurryTimestampList, dim=0) # LxNumPx2
        SeqExposureDuty = torch.stack(SeqExposureDutyList, dim=0) # LxNumPx1

        if self.config['data_augment']['enabled']:
            SeqLatentF = self.AugmentData(SeqLatentF, 'frame', seed)
            SeqBlurryF = self.AugmentData(SeqBlurryF, 'frame',seed)
            if self.NeedNeighborGT:
                SeqNeighborF = self.AugmentData(SeqNeighborF, 'frame',seed)
            # SeqLREv = self.AugmentData(SeqLREv, 'LRevent', seed)
            # SeqLREvBi = self.AugmentData(SeqLREvBi, 'LReventBi', seed)
            SeqHREv = self.AugmentData(SeqHREv, 'HRevent', seed)

        if self.NeedNeighborGT:
            return {
                'SeqLatentF': SeqLatentF, # LxNumPxNumFx3xHxW or LxNumPx1x3xHxW
                'SeqBlurryF': SeqBlurryF, # LxNumPx3xHxW
                'SeqNeighborF': SeqNeighborF, # LxNumPxNumFx2x3xHxW
                # 'SeqLREv': SeqLREv, # LxTBx2xHxW
                # 'SeqLREvBi': SeqLREvBi, # LxTBx2xHxW
                'SeqHREv': SeqHREv, # LxTBx2xHxW

                'LatentTs': LatentTs, # Lx(NumP*NumF)
                'RelativeLatentTs': RelativeLatentTs, # LxNumPx(NumP*NumF)
                'BlurryTs': BlurryTs, # LxNumPx2

                'SeqExposureDuty': SeqExposureDuty, # LxNumPx1
            }
        else:
            return {
                'SeqLatentF': SeqLatentF, # LxNumPxNumFx3xHxW or LxNumPx1x3xHxW
                'SeqBlurryF': SeqBlurryF, # LxNumPx3xHxW
                # 'SeqLREv': SeqLREv, # LxTBx2xHxW
                # 'SeqLREvBi': SeqLREvBi, # LxTBx2xHxW
                'SeqHREv': SeqHREv, # LxTBx2xHxW

                'LatentTs': LatentTs, # Lx(NumP*NumF)
                'RelativeLatentTs': RelativeLatentTs, # LxNumPx(NumP*NumF)
                'BlurryTs': BlurryTs, # LxNumPx2

                'SeqExposureDuty': SeqExposureDuty, # LxNumPx1
            }

    def GetFrames(self, Indices, mode):
        FList = []
        if mode != 'neighbor':
            if self.DeblurPretrain and mode == 'sharp':
                Indices = [Indices[-1]]
            for i in Indices:
                frame = self.h5_file['ori_images'][f'image{i:09d}'][:][:, :, (2, 1, 0)] # BGR --> RGB
                if frame.shape[:-1] != self.gt_sensor_resolution:
                    frame = cv2.resize(frame, self.gt_sensor_resolution[::-1], interpolation=cv2.INTER_CUBIC) # H1xW1x3 --> HxWx3
                FList.append(frame)
            FNumpy = np.stack(FList) # NxHxWx3
            if mode == 'sharp':
                return torch.from_numpy(FNumpy).permute(0, 3, 1, 2).float() / 255 # Nx3xHxW
            elif mode == 'blurry':
                return torch.from_numpy(FNumpy.mean(0)).permute(2, 0, 1).float() / 255 # 3xHxW
            else:
                raise Exception('Error mode!')
        else:
            for IList in Indices:
                NeiList = []
                for i in IList:
                    frame = self.h5_file['ori_images'][f'image{i:09d}'][:][:, :, (2, 1, 0)] # BGR --> RGB
                    if frame.shape[:-1] != self.gt_sensor_resolution:
                        frame = cv2.resize(frame, self.gt_sensor_resolution[::-1], interpolation=cv2.INTER_CUBIC) # H1xW1x3 --> HxWx3
                    NeiList.append(frame)
                NeiFNumpy = np.stack(NeiList)
                NeiF = torch.from_numpy(NeiFNumpy).permute(0, 3, 1, 2).float() / 255 # 2x3xHxW
                FList.append(NeiF)
            return torch.stack(FList) # Nx2x3xHxW

    def GetEvents(self, Indices):
        def GetEventsIndex(idx0, idx1, prex):
            xs = self.h5_file[f'{prex}_events/xs'][idx0:idx1]
            ys = self.h5_file[f'{prex}_events/ys'][idx0:idx1]
            ts = self.h5_file[f'{prex}_events/ts'][idx0:idx1]
            ps = self.h5_file[f'{prex}_events/ps'][idx0:idx1]
            if len(xs) == 0 or len(ys) == 0 or len(ts) == 0 or len(ps) == 0:
                xs = ys = ts = ps = np.array([0.])
            ts = (ts - ts[0]) / (ts[-1] - ts[0] + 1e-6)
            return torch.from_numpy(np.concatenate((xs[np.newaxis, ...], ys[np.newaxis, ...], ts[np.newaxis, ...], ps[np.newaxis, ...]), axis=0))

        start, end = Indices[0], Indices[-1]
        # LREvLeftIndex = self.h5_file['ori_images'][f'image{start:09d}'].attrs[f'{self.inp_prex}_event_idx']
        # LREvRightIndex = self.h5_file['ori_images'][f'image{end:09d}'].attrs[f'{self.inp_prex}_event_idx']
        HREvLeftIndex = self.h5_file['ori_images'][f'image{start:09d}'].attrs[f'{self.gt_prex}_event_idx']
        HREvRightIndex = self.h5_file['ori_images'][f'image{end:09d}'].attrs[f'{self.gt_prex}_event_idx']

        # LREvents = GetEventsIndex(LREvLeftIndex, LREvRightIndex, self.inp_prex) # 4xN
        HREvents = GetEventsIndex(HREvLeftIndex, HREvRightIndex, self.gt_prex) # 4xN

        # LRStack = events_to_stack(xs=LREvents[0], ys=LREvents[1], ts=LREvents[2], ps=LREvents[3].float(), B=self.time_bins, sensor_size=self.inp_sensor_resolution).transpose(0, 1) # TBx2xH1xW1
        # LRStackBi = func.interpolate(LRStack, size=self.gt_sensor_resolution, mode='bicubic', align_corners=True).abs().round().float() # TBx2xHxW
        HRStack = events_to_stack(xs=HREvents[0], ys=HREvents[1], ts=HREvents[2], ps=HREvents[3].float(), B=self.time_bins, sensor_size=self.gt_sensor_resolution).transpose(0, 1) # TBx2xHxW

        # return LRStack, LRStackBi, HRStack
        return HRStack

    def GetTimestamp(self, LatentTs, BlurryTs):
        t0 = LatentTs[0]
        BlurryTsTmp = []
        for item in BlurryTs:
            BlurryTsTmp.append([item[0], item[-1]])
        LatentTs, BlurryTs = (torch.tensor(LatentTs) - t0) / self.Interval, (torch.tensor(BlurryTsTmp) - t0) / self.Interval
        RelativeTsList = [LatentTs]
        for i in range(1, self.NumPeriodPerLoad):
            ts = 1 / (i + 1)
            RelativeTsList.append(LatentTs - ts)
        RelativeTs = torch.stack(RelativeTsList)

        return LatentTs, RelativeTs, BlurryTs

    def AugmentData(self, data, type, seed):
        def random_crop(x, output_size, scale, seed):
            # w, h = x.size()[-1], x.size()[-2]
            w, h = self.gt_sensor_resolution[-1], self.gt_sensor_resolution[-2]
            th, tw = output_size
            if th >= h or tw >= w:
            #     raise Exception("Input size {}x{} is less than desired cropped \
            #             size {}x{} - input tensor shape = {}".format(w, h, tw, th, x.shape))
            # if w == tw and h == th:
                return x

            random.seed(seed)
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            i = int(i // scale) * scale
            j = int(j // scale) * scale

            i, j, th, tw = i // scale, j // scale, th // scale, tw // scale

            return x[..., i:i + th, j:j + tw]
        
        def center_crop(x, output_size, scale, seed):
            # w, h = x.size()[-1], x.size()[-2]
            w, h = self.gt_sensor_resolution[-1], self.gt_sensor_resolution[-2]
            th, tw = output_size
            if th >= h or tw >= w:
            #     raise Exception("Input size {}x{} is less than desired cropped \
            #             size {}x{} - input tensor shape = {}".format(w, h, tw, th, x.shape))
            # if w == tw and h == th:
                return x

            i = int((h - th) / 2)
            j = int((w - tw) / 2)
            i = int(i // scale) * scale
            j = int(j // scale) * scale

            i, j, th, tw = i // scale, j // scale, th // scale, tw // scale

            return x[..., i:i + th, j:j + tw]

        assert type in ['frame', 'LRevent', 'LReventBi', 'HRevent'], 'Error type!'
        seed_Flip_H, seed_Flip_V, seed_Crop, seed_noise, seed_hotpixel = seed, seed + 1, seed + 2, seed + 3, seed + 4

        for i, mechanism in enumerate(self.config['data_augment']['augment']):
            if mechanism == 'HorizontalFlip':
                if self.config['data_augment']['flip']['enabled']:
                    random.seed(seed_Flip_H)
                    if random.random() < self.config['data_augment']['flip']['horizontal_prob']:
                        data = data.flip(-1)
            elif mechanism == 'VertivcalFlip':
                if self.config['data_augment']['flip']['enabled']:
                    random.seed(seed_Flip_V)
                    if random.random() < self.config['data_augment']['flip']['vertical_prob']:
                        data = data.flip(-2)
            elif mechanism == 'RandomCrop':
                if self.config['data_augment']['random_crop']['enabled']:
                    if type in ['LReventBi', 'HRevent', 'frame']:
                        data = random_crop(data, self.config['data_augment']['random_crop']['size'], 1, seed_Crop)
                    elif type == 'LRevent':
                        data = random_crop(data, self.config['data_augment']['random_crop']['size'], self.scale, seed_Crop)
            elif mechanism == 'CenterCrop':
                if self.config['data_augment']['center_crop']['enabled']:
                    if type in ['LReventBi', 'HRevent', 'frame']:
                        data = center_crop(data, self.config['data_augment']['center_crop']['size'], 1, seed_Crop)
                    elif type == 'LRevent':
                        data = center_crop(data, self.config['data_augment']['center_crop']['size'], self.scale, seed_Crop)
            elif mechanism == 'Noise':
                if type in ['LRevent', 'LReventBi', 'HRevent'] and self.config['data_augment']['noise']['enabled']:
                    data = self.add_noise(data, seed_noise, self.config['data_augment']['noise']['noise_std'], self.config['data_augment']['noise']['noise_fraction'])
            elif mechanism == 'HotPixel':
                if type == ['LRevent', 'LReventBi', 'HRevent'] and self.config['data_augment']['hot_pixel']['enabled']:
                    self.add_hot_pixels(data, seed_hotpixel, self.config['data_augment']['hot_pixel']['hot_pixel_std'], self.config['data_augment']['hot_pixel']['hot_pixel_fraction'])
            else:
                raise Exception('Error augmentation!')

        return data

    @staticmethod
    def add_hot_pixels(data, seed, hot_pixel_std=1.0, hot_pixel_fraction=0.001):
        torch.manual_seed(seed)
        num_hot_pixels = int(hot_pixel_fraction * data.shape[-1] * data.shape[-2])
        x = torch.randint(0, data.shape[-1], (num_hot_pixels,))
        y = torch.randint(0, data.shape[-2], (num_hot_pixels,))
        for i in range(num_hot_pixels):
            data[..., :, y[i], x[i]] += (hot_pixel_std*torch.randn(1)).abs().int()
    
    @staticmethod
    def add_noise(data, seed, noise_std=1.0, noise_fraction=0.1):
        torch.manual_seed(seed)
        noise = (noise_std * torch.randn_like(data)).abs().int()  # mean = 0, std = noise_std
        if noise_fraction < 1.0:
            mask = torch.rand_like(data) >= noise_fraction
            noise.masked_fill_(mask, 0)

        return data + noise 


if __name__ == '__main__':
    import yaml
    import time
    from myutils.vis_events.matplotlib_plot_events import event_visualisation

    vis = event_visualisation()

    path_to_h5 = '/data2/wengwm/work/dataset/adobe240/h5/720p_240fps_3.h5'
    path_to_config = '/data2/wengwm/work/code/ESRVFI/config/train_ledvdi.yml'
    path_to_output = '/data2/wengwm/work/output/data_check/720p_240fps_3-down8'
    with open(path_to_config) as fid:
        yaml_config = yaml.load(fid, Loader=yaml.FullLoader)
    dataset_config = yaml_config['train_dataloader']['dataset']
    dataset = H5Dataset(path_to_h5, dataset_config)
    length = dataset.__len__()
    t0 = time.time()
    for ix in range(length):
        print(ix)
        item = dataset.__getitem__(ix)
        print(item['SeqLatentF'].size()) 
    t1 = time.time()
    print(f'time: {t1-t0}')
        # path = os.path.join(path_to_output, f'{ix}')
        # path_to_latent = os.path.join(path, 'latent')
        # path_to_blurry = os.path.join(path, 'blurry')
        # path_to_lr_event = os.path.join(path, 'lr_event')
        # path_to_lrbi_event = os.path.join(path, 'lrbi_event')
        # path_to_hr_event = os.path.join(path, 'hr_event')
        # os.makedirs(path_to_latent)
        # os.makedirs(path_to_blurry)
        # os.makedirs(path_to_lr_event)
        # os.makedirs(path_to_lrbi_event)
        # os.makedirs(path_to_hr_event)

        # item = dataset.__getitem__(ix)
        # SeqLatentF = item['SeqLatentF'] # LxNumPxNumFx3xHxW 
        # SeqBlurryF = item['SeqBlurryF'] # LxNumPx3xHxW
        # SeqLREv = item['SeqLREv'] # LxTBx2xHxW
        # SeqLREvBi = item['SeqLREvBi'] # LxTBx2xHxW
        # SeqHREv = item['SeqHREv'] # LxTBx2xHxW
        # LatentTs = item['LatentTs'] # Lx(NumP*NumF)
        # RelativeLatentTs = item['RelativeLatentTs'] # LxNumPx(NumP*NumF)
        # BlurryTs = item['BlurryTs'] # LxNumPx2

        # SeqL = SeqLatentF.size()[0]
        # LoadL = SeqLatentF.size()[1]
        # FrameL = SeqLatentF.size()[2]
        # BinsL = SeqLREv.size()[1]
        # for i in range(SeqL):
        #     os.makedirs(os.path.join(path_to_latent, f'{i}'))
        #     os.makedirs(os.path.join(path_to_blurry, f'{i}'))
        #     os.makedirs(os.path.join(path_to_lr_event, f'{i}'))
        #     os.makedirs(os.path.join(path_to_lrbi_event, f'{i}'))
        #     os.makedirs(os.path.join(path_to_hr_event, f'{i}'))
        #     for tb in range(BinsL):
        #         vis.plot_event_cnt(SeqLREv[i, tb].round().numpy().transpose(1, 2, 0),
        #             is_save=True,
        #             path=os.path.join(path_to_lr_event, f'{i}', f'{tb:06d}.png'),
        #             color_scheme="blue_red", is_black_background=False, is_norm=True
        #         )
        #         vis.plot_event_cnt(SeqLREvBi[i, tb].round().numpy().transpose(1, 2, 0),
        #             is_save=True,
        #             path=os.path.join(path_to_lrbi_event, f'{i}', f'{tb:06d}.png'),
        #             color_scheme="blue_red", is_black_background=False, is_norm=True
        #         )
        #         vis.plot_event_cnt(SeqHREv[i, tb].round().numpy().transpose(1, 2, 0),
        #             is_save=True,
        #             path=os.path.join(path_to_hr_event, f'{i}', f'{tb:06d}.png'),
        #             color_scheme="blue_red", is_black_background=False, is_norm=True
        #         )
        #     for j in range(LoadL):
        #         os.makedirs(os.path.join(path_to_latent, f'{i}', f'{j}'))
        #         os.makedirs(os.path.join(path_to_blurry, f'{i}', f'{j}'))
        #         vis.plot_frame((SeqBlurryF[i, j].numpy().transpose(1, 2, 0) * 255).astype('uint8'),
        #                 is_save=True,
        #                 path=os.path.join(path_to_blurry, f'{i}', f'{j}', 'blurry.png')
        #         )
        #         for k in range(FrameL):
        #             vis.plot_frame((SeqLatentF[i, j, k].numpy().transpose(1, 2, 0) * 255).astype('uint8'),
        #                 is_save=True,
        #                 path=os.path.join(path_to_latent, f'{i}', f'{j}', f'latent_{k}.png')
        #             )


    