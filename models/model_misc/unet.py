"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import torch
import torch.nn as nn

from .model_util import *
from .submodules import (
    ConvGRU,
    ConvLayer,
    RecurrentConvLayer,
    ResidualBlock,
    TransposedConvLayer,
    UpsampleConvLayer,
)


class BaseUNet(nn.Module):
    """
    Base class for conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(
        self,
        base_num_channels,
        num_encoders,
        num_residual_blocks,
        num_output_channels,
        skip_type,
        norm,
        use_upsample_conv,
        num_bins,
        recurrent_block_type=None,
        kernel_size=5,
        channel_multiplier=2,
    ):
        super(BaseUNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type
        self.channel_multiplier = channel_multiplier

        self.skip_ftn = eval("skip_" + skip_type)
        if use_upsample_conv:
            self.UpsampleLayer = UpsampleConvLayer
        else:
            self.UpsampleLayer = TransposedConvLayer
        assert self.num_output_channels > 0

        self.encoder_input_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i)) for i in range(self.num_encoders)
        ]
        self.encoder_output_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i + 1)) for i in range(self.num_encoders)
        ]
        self.max_num_channels = self.encoder_output_sizes[-1]

    def build_encoders(self):
        encoders = nn.ModuleList()
        for (input_size, output_size) in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            encoders.append(
                ConvLayer(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.kernel_size // 2,
                    activation=self.activation,
                    norm=self.norm,
                )
            )
        return encoders

    def build_resblocks(self):
        resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))
        return resblocks

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(
                self.UpsampleLayer(
                    input_size if self.skip_type == "sum" else 2 * input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    norm=self.norm,
                )
            )
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(
            self.base_num_channels if self.skip_type == "sum" else 2 * self.base_num_channels,
            num_output_channels,
            1,
            activation=None,
            norm=norm,
        )


class BaseUNet_legacy(nn.Module):
    """
    ECCV20
    """
    def __init__(self, base_num_channels, num_encoders, num_residual_blocks,
                 num_output_channels, skip_type, norm, use_upsample_conv,
                 num_bins, recurrent_block_type=None, kernel_size=5,
                 channel_multiplier=2):
        super(BaseUNet_legacy, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type

        self.encoder_input_sizes = [int(self.base_num_channels * pow(channel_multiplier, i)) for i in range(self.num_encoders)]
        self.encoder_output_sizes = [int(self.base_num_channels * pow(channel_multiplier, i + 1)) for i in range(self.num_encoders)]
        self.max_num_channels = self.encoder_output_sizes[-1]
        self.skip_ftn = eval('skip_' + skip_type)
        print('Using skip: {}'.format(self.skip_ftn))
        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer
        assert(self.num_output_channels > 0)
        print(f'Kernel size {self.kernel_size}')
        print(f'Skip type {self.skip_type}')
        print(f'norm {self.norm}')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(self.UpsampleLayer(
                input_size if self.skip_type == 'sum' else 2 * input_size,
                output_size, kernel_size=self.kernel_size,
                padding=self.kernel_size // 2, norm=self.norm))
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                         num_output_channels, 1, activation=None, norm=norm)


class UNetFlow(BaseUNet_legacy):
    """
    ECCV20
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        unet_kwargs['num_output_channels'] = 3
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(num_output_channels=3)
        self.states = [None] * self.num_encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img_flow = self.pred(self.skip_ftn(x, head))

        output_dict = {'image': img_flow[:, 0:1, :, :], 'flow': img_flow[:, 1:3, :, :]}

        return output_dict


class UNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        final_activation = unet_kwargs.pop("final_activation", "none")
        self.final_activation = getattr(torch, final_activation, None)
        unet_kwargs["num_output_channels"] = 1
        super().__init__(**unet_kwargs)

        self.head = ConvLayer(
            self.num_bins,
            self.base_num_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )

        self.encoders = self.build_recurrent_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = [None] * self.num_encoders

    def build_recurrent_encoders(self):
        encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            encoders.append(
                RecurrentConvLayer(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.kernel_size // 2,
                    recurrent_block_type=self.recurrent_block_type,
                    norm=self.norm,
                )
            )
        return encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)
        return img


class MultiResUNet(BaseUNet):
    """
    Conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """

    def __init__(self, unet_kwargs):
        self.final_activation = unet_kwargs.pop("final_activation", "none")
        self.skip_type = "concat"
        super().__init__(**unet_kwargs)

        self.encoders = self.build_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()

    def build_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                ConvLayer(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.kernel_size // 2,
                    norm=self.norm,
                )
            )
        return encoders

    def build_multires_prediction_layer(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for output_size in decoder_output_sizes:
            preds.append(
                ConvLayer(output_size, self.num_output_channels, 1, activation=self.final_activation, norm=self.norm)
            )
        return preds

    def build_multires_prediction_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            decoders.append(
                self.UpsampleLayer(
                    2 * input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    norm=self.norm,
                )
            )
        return decoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, blocks[self.num_encoders - i - 1])
            if i > 0:
                x = self.skip_ftn(predictions[-1], x)
            x = decoder(x)
            predictions.append(pred(x))

        return predictions


class SRUNetRecurrent(BaseUNet):
    """
    SR Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        final_activation = unet_kwargs.pop("final_activation", "none")
        self.final_activation = getattr(torch, final_activation, None)
        # unet_kwargs["num_output_channels"] = 1
        super().__init__(**unet_kwargs)

        self.head = ConvLayer(
            self.num_bins,
            self.base_num_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )

        self.encoders = self.build_recurrent_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_SRdecoders()
        self.skip_upsampler = self.build_skip_upsampler()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = [None] * self.num_encoders

    def build_recurrent_encoders(self):
        encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            encoders.append(
                RecurrentConvLayer(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.kernel_size // 2,
                    recurrent_block_type=self.recurrent_block_type,
                    norm=self.norm,
                )
            )
        return encoders

    def build_SRdecoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        SRdecoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            SRdecoders.append(
                self.UpsampleLayer(
                    input_size if self.skip_type == "sum" else 2 * input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    norm=self.norm,
                    scale=4 if i==0 else 2
                )
            )
        return SRdecoders

    def build_skip_upsampler(self):
        skip_sizes = self.encoder_output_sizes[::-1] + [self.base_num_channels]
        skip_upsampler = nn.ModuleList()
        for input_size in skip_sizes:
            skip_upsampler.append(
                self.UpsampleLayer(
                    input_size if self.skip_type == "sum" else 2 * input_size,
                    input_size,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    norm=self.norm,
                    scale=2
                )
            )
        return skip_upsampler

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x 2H x 2W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, self.skip_upsampler[i](blocks[self.num_encoders - i - 1])))

        # tail
        img = self.pred(self.skip_ftn(x, self.skip_upsampler[-1](head)))
        if self.final_activation is not None:
            img = self.final_activation(img)
        return img


if __name__ == '__main__':
    input = torch.ones([2, 5, 8, 8])
    unet_kwargs = {
        'base_num_channels': 32,
        'num_encoders': 3,
        'num_residual_blocks': 2,
        'num_output_channels': 5,
        'skip_type': 'sum',
        'norm': None,
        'use_upsample_conv': True,
        'num_bins': 5,
        'recurrent_block_type': 'convlstm',
        'kernel_size': 5,
    }

    model = SRUNetRecurrent(unet_kwargs)

    output = model(input)

    print(input.size())
    print(output.size())