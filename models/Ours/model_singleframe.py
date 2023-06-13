import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange, reduce, repeat
from math import sqrt
from torchvision.models.resnet import resnet34

from myutils.utils import Frame2DCP, Frame2Lap
from ..model_misc.base import BaseModel
from ..model_misc.submodules import *
from ..model_misc.model_util import *
from myutils.vis_events.matplotlib_plot_events import *
from models.DCNv2.dcn_v2 import DCN_sep
from models.FAC.kernelconv2d.KernelConv2D import KernelConv2D


###############################################################################
####ESRVFI#####################################################################
###############################################################################
class ExposureDecision(BaseModel):
    def __init__(self, EventInch=32, BLInch=1, InterCH=64, Group=4, norm=None, activation='LeakyReLU', LoadPretrain=False, PretrainedEXPath=None, Frozen=False):
        super().__init__()
        self.LoadPretrain = LoadPretrain
        self.PretrainedEXPath = PretrainedEXPath
        self.Frozen = Frozen

        self.EventFeatExtract = ConvLayer(in_channels=EventInch, out_channels=InterCH, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation)
        self.BLFeatExtract = ConvLayer(in_channels=BLInch, out_channels=InterCH, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation)

        self.GroupNorm = nn.GroupNorm(Group, InterCH)
        self.AVGPool = nn.AdaptiveAvgPool2d(1)

        self.Conv1 = nn.Sequential(
                ConvLayer(in_channels=2*InterCH, out_channels=InterCH, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
                ConvLayer(in_channels=InterCH, out_channels=1, kernel_size=3, stride=1, padding=1, norm=norm, activation=None),
            )

        initialize_weights([self.EventFeatExtract, self.BLFeatExtract, self.GroupNorm, self.Conv1], 0.1)
        self.load_pretrain()

    def load_pretrain(self):
        if self.LoadPretrain:
            # print('Load pretrained ex!')
            # cpt_path = '/data2/wengwm/work/output3/TrainExposureDecision/models/EVFIAutoEx/exposuredecision-adobe240-1gpu/model_best_until_iteration75000.pth'
            cpt = torch.load(self.PretrainedEXPath, map_location='cpu')
            self.load_state_dict(cpt['model']['states'])

        if self.Frozen:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def forward(self, Event, BlurryLevel):
        '''
        Event: Bx(TBx2)xHxW
        BlurryLevel: Bx1xHxW

        return: ExOut: Bx1, [0, 1]
        '''
        EventFeat = self.EventFeatExtract(Event) # BxCxH1xW1
        BLFeat = self.BLFeatExtract(BlurryLevel) # BxCxH1xW1

        EventNorm = self.GroupNorm(EventFeat) # BxCxH1xW1
        BLNorm = self.GroupNorm(BLFeat) # BxCxH1xW1
        Corre = EventNorm * BLNorm # BxCxH1xW1
        Atten = torch.sigmoid(self.AVGPool(Corre)) # BxCx1x1

        EventFeatSelected = EventFeat * Atten # BxCxH1xW1

        ExOut = self.Conv1(torch.cat([EventFeatSelected, BLFeat], dim=1)) # Bx1xH1xW1
        ExOut = torch.sigmoid(self.AVGPool(ExOut).view(-1, 1)) # Bx1
        
        return ExOut


class ResidualControl(BaseModel):
    def __init__(self, BLinch=2, Tinch=1, Basech=16, step=4, norm=None, activation='LeakyReLU'):
        super().__init__()
        self.step = step

        self.Conv1 = nn.ModuleList([
                nn.Sequential(
                ConvLayer(in_channels=BLinch, out_channels=Basech, kernel_size=1, stride=1, padding=0, norm=norm, activation=activation),
            ) for _ in range(step)
        ])
        self.Conv2 = nn.ModuleList([
                nn.Sequential(
                ConvLayer(in_channels=Tinch, out_channels=Basech, kernel_size=1, stride=1, padding=0, norm=norm, activation=activation),
            ) for _ in range(step)
        ])
        self.Conv3 = nn.ModuleList([
                nn.Sequential(
                ConvLayer(in_channels=Basech, out_channels=Basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
                ConvLayer(in_channels=Basech, out_channels=Basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
            ) for _ in range(step)
        ])
        self.Conv4 = nn.ModuleList([
                nn.Sequential(
                ConvLayer(in_channels=Basech, out_channels=Basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
                ConvLayer(in_channels=Basech, out_channels=Basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
            ) for _ in range(step)
        ])
        self.Conv5 = nn.ModuleList([
                nn.Sequential(
                ConvLayer(in_channels=2*Basech, out_channels=Basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
            ) for _ in range(step)
        ])

        # initialization
        initialize_weights([self.Conv1, self.Conv2, self.Conv3, self.Conv4, self.Conv5], 0.1)

    def forward(self, data, Ex, T):
        '''
        data: torch.tensor, BxCxHxW
        Ex: torch.tensor, Bx1
        T: torch.tensor, Bx1

        return: torch.tensor, BxCxHxW
        '''
        Ex = Ex.unsqueeze(-1).unsqueeze(-1) # Bxn1x1x1
        T = T.unsqueeze(-1).unsqueeze(-1) # Bxnx1x1

        inp = data
        for i in range(self.step):
            ExScale = self.Conv1[i](Ex) # BxCx1x1
            TScale = self.Conv2[i](T) # BxCx1x1
            Exx = self.Conv3[i](inp) # BxCxHxW
            Tx = self.Conv4[i](inp) # BxCxHxW
            ExOut = ExScale * Exx + inp # BxCxHxW
            TOut = TScale * Tx + inp # BxCxHxW
            inp = self.Conv5[i](torch.cat([ExOut, TOut], dim=1)) # BxCxHxW
        
        return inp


class Modification(BaseModel):
    def __init__(self, FrameBasech=64, EventBasech=32, TB=16, KernelSize=5, norm=None, activation='LeakyReLU'):
        super().__init__()

        self.Conv1 = ConvLayer(in_channels=EventBasech, out_channels=FrameBasech, kernel_size=1, stride=1, padding=0, norm=norm, activation=activation)
        self.Conv2 = ConvLayer(in_channels=FrameBasech, out_channels=FrameBasech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation)
        self.KernelConv = ConvLayer(in_channels=FrameBasech+FrameBasech, out_channels=FrameBasech*KernelSize**2, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation)
        self.KPN = KernelConv2D(kernel_size=KernelSize)
        self.Conv3 = ConvLayer(in_channels=FrameBasech, out_channels=FrameBasech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation)

        # initialization
        initialize_weights([self.Conv1, self.Conv2, self.Conv3, self.KernelConv], 0.1)

    def forward(self, FrameTensor, EventTensor):
        '''
        FrameTensor: torch.tensor, BxC1xHxW
        EventTensor: torch.tensor, BxC2xHxW

        return: SharpTensor, BxC1xHxW
        '''
        EventTensor = self.Conv1(EventTensor) # BxC2xHxW -> BxC1xHxW

        Kernel = self.KernelConv(torch.cat([EventTensor, FrameTensor], dim=1))
        EventTensor1 = self.Conv3(self.KPN(EventTensor, Kernel))
        SharpTensor = FrameTensor * EventTensor1 + self.Conv2(EventTensor1)

        return SharpTensor



from ..model_misc.resnet_3D import r3d_18, Conv_3d, upConv3D, identity
class UNet3d_18(nn.Module):
    def __init__(self, channels=[32,64,96,128], bn=True):
        super(UNet3d_18, self).__init__()
        growth = 2 # since concatenating previous outputs
        upmode = "transpose" # use transposeConv to upsample

        self.channels = channels

        self.lrelu = nn.LeakyReLU(0.2, True)

        self.encoder = r3d_18(bn=bn, channels=channels)

        self.decoder = nn.Sequential(
            Conv_3d(channels[::-1][0], channels[::-1][1] , kernel_size=3, padding=1, bias=True),
            upConv3D(channels[::-1][1]*growth, channels[::-1][2], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode),
            upConv3D(channels[::-1][2]*growth, channels[::-1][3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode),
            Conv_3d(channels[::-1][3]*growth, channels[::-1][3] , kernel_size=3, padding=1, bias=True),
            upConv3D(channels[::-1][3]*growth , channels[::-1][3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode)
        )

        self.feature_fuse = nn.Sequential(
            *([nn.Conv2d(channels[::-1][3]*2, channels[::-1][3], kernel_size=1, stride=1, bias=False)] + \
              [nn.BatchNorm2d(channels[::-1][3]) if bn else identity()])
        )

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels[::-1][3], 3 , kernel_size=7, stride=1, padding=0) 
        )
    
    def forward(self, img0, img1):
        images = torch.stack((img0, img1) , dim=2)

        x_0 , x_1 , x_2 , x_3 , x_4 = self.encoder(images)

        dx_3 = self.lrelu(self.decoder[0](x_4))
        dx_3 = torch.cat([dx_3 , x_3], dim=1)

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        dx_2 = torch.cat([dx_2 , x_2], dim=1)

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        dx_1 = torch.cat([dx_1 , x_1], dim=1)

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        dx_0 = torch.cat([dx_0 , x_0], dim=1)

        dx_out = self.lrelu(self.decoder[4](dx_0))
        dx_out = torch.cat(torch.unbind(dx_out , 2) , 1)

        out = self.lrelu(self.feature_fuse(dx_out))
        out = self.outconv(out)

        return out


class EVFIAutoEx(BaseModel):
    def __init__(self, FrameBasech=64, EventBasech=64, InterCH=64, TB=16, norm=None, activation='LeakyReLU', 
                        # exposure decision
                        BlurryFashion='DarkCh', BLInch=1, UseEvents=True, UseGTEx=False, FixEx=None, LoadPretrainEX=False, PretrainedEXPath=None, FrozenEX=False,
                        # time-exposure control
                        step=32, DualPath=True,
                        # modification
                        residual=True,
                        # detail restoration
                        DetailEnabled=True, channels=[32, 64, 96, 128], 
                        ):
        super().__init__()
        self.TB = TB
        self.UseGTEx = UseGTEx
        self.FixEx = FixEx
        self.BlurryFashion = BlurryFashion
        self.DetailEnabled = DetailEnabled

        self.FrameFeatExtract = ConvLayer(in_channels=3, out_channels=FrameBasech, kernel_size=3, stride=2, padding=1, norm=norm, activation=activation)
        self.EventFeatExtract = ConvLayer(in_channels=2*TB, out_channels=EventBasech, kernel_size=3, stride=2, padding=1, norm=norm, activation=activation)

        if not self.UseGTEx and not self.FixEx:
            if UseEvents:
                self.ExposureDecision = ExposureDecision(EventInch=2*TB, BLInch=BLInch, InterCH=InterCH, Group=4, norm=norm, activation=activation, LoadPretrain=LoadPretrainEX, PretrainedEXPath=PretrainedEXPath, Frozen=FrozenEX)
        
        if DualPath:
            self.ResidualControl = ResidualControl(BLinch=1, Tinch=1, Basech=EventBasech, step=step, norm=norm, activation=activation)
        
        if residual:
            self.Modification = Modification(FrameBasech=FrameBasech, EventBasech=EventBasech, TB=TB, KernelSize=5, norm=norm, activation=activation)

        self.Reconstruction = nn.Sequential(
            nn.Sequential(
            ConvLayer(in_channels=FrameBasech, out_channels=FrameBasech*4, kernel_size=3, stride=1, padding=1, norm=norm, activation=None),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True),
        ),
            ConvLayer(in_channels=FrameBasech, out_channels=FrameBasech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
            ConvLayer(in_channels=FrameBasech, out_channels=3, kernel_size=3, stride=1, padding=1, norm=norm, activation='Sigmoid'),
            # ConvLayer(in_channels=FrameBasech, out_channels=3, kernel_size=3, stride=1, padding=1, norm=norm, activation=None),
        )

        if self.DetailEnabled:
            self.Detail = UNet3d_18(channels=channels, bn=False)

        # initialization
        initialize_weights([self.FrameFeatExtract, self.EventFeatExtract, self.Reconstruction], 0.1)

    def LoadExposureDecision(self):
        self.ExposureDecision.load_pretrain()

    def forward(self, Frame, Event, T, GTEx=None):
        '''
        Frame: torch.tensor, Bx3xHxW
        Event: torch.tensor, BxTBx2xHxW
        T: torch.tensor, Bx1

        return: sharp, Bx3xHxW
        '''
        #################################################
        ##No multi-scale
        #################################################
        # pad input
        H, W = Frame.size()[-2:]
        factor = {'h': 8, 'w': 8}
        need_crop = (H % factor['h'] != 0) or (W % factor['w'] != 0)
        pad_crop = CropSize(W, H, factor) if need_crop else None
        if need_crop and pad_crop:
            Frame = pad_crop.pad(Frame)
            Event = pad_crop.pad(Event)

        Event = Event.view(Event.size(0), -1, Event.size(3), Event.size(4)) # Bx(TBx2)xHxW

        FrameFeat = self.FrameFeatExtract(Frame) # frame: BxC1xHxW
        EventFeat = self.EventFeatExtract(Event) # event: BxC2xHxW

        if self.UseGTEx:
            assert self.FixEx is None, 'set UseGTEx, but FixEx is given!'
            assert GTEx is not None, 'set UseGTEx, but NO GTEx provided!'
            Ex = GTEx
        elif self.FixEx:
            assert self.UseGTEx is False, 'set FixEx, but UseGTEx is set!'
            assert self.FixEx <= 1 and self.FixEx>=0, 'Wrong FixEx!'
            Ex = torch.tensor(self.FixEx).unsqueeze(0).repeat(Frame.size(0), 1).cuda()
        else:
            if self.BlurryFashion == 'DarkCh':
                BlurryLevel = Frame2DCP(Frame) # Bx1xHxW
            elif self.BlurryFashion == 'Lap':
                BlurryLevel = Frame2Lap(Frame) # Bx1xHxW
            elif self.BlurryFashion == 'RGB':
                BlurryLevel = Frame # Bx3xHxW

            elif self.BlurryFashion == 'RGBDark':
                Dark = Frame2DCP(Frame) # Bx1xHxW
                BlurryLevel = torch.cat([Frame, Dark], dim=1) # Bx4xHxW
            elif self.BlurryFashion == 'RGBLap':
                Lap = Frame2Lap(Frame) # Bx1xHxW
                BlurryLevel = torch.cat([Frame, Lap], dim=1) # Bx4xHxW
            else:
                raise Exception('Wrong blurry convertion fashion!!')
            Ex = self.ExposureDecision(Event, BlurryLevel) # Bx1

        ProcessedEventFeat = self.ResidualControl(EventFeat, Ex, T) # BxC2xHxW
        ProcessedFrameFeat = self.Modification(FrameFeat, ProcessedEventFeat) # BxC1xHxW

        Sharp = self.Reconstruction(ProcessedFrameFeat) # Bx3xHxW

        if self.DetailEnabled:
            Detail = self.Detail(img0=Frame, img1=Sharp) # Bx3xHxW
            Final = Sharp + Detail

        # crop output
        if need_crop and pad_crop:
            if self.DetailEnabled:
                Final = pad_crop.crop(Final)
                Final = Final.contiguous()
            Sharp = pad_crop.crop(Sharp)
            Sharp = Sharp.contiguous()

        if self.DetailEnabled:
            return Sharp, Final
        else:
            return Sharp, Sharp


if __name__ == '__main__':
    esrvfi_config = {
        'FrameBasech': 64,
        'EventBasech': 64,
        'InterCH': 64,
        'TB': 16,
        'step': 8,
        'norm': None,
        'activation': 'LeakyReLU'
    }

    B = 2
    TB = 16
    H, W = 128, 128

    device = torch.device('cuda')
    net = EVFIAutoEx(**esrvfi_config).to(device)

    print(net)

    LeftFrame = torch.randn([B, 3, H, W]).to(device)
    RightFrame = torch.randn([B, 3, H, W]).to(device)
    Event = torch.randn([B, TB, 2, H, W]).to(device)
    LeftT = torch.randn([B, 1]).to(device)
    RightT = torch.randn([B, 1]).to(device)

    while True:
        out = net(LeftFrame, Event, LeftT)
        print(out.size())