import torch
import torch.nn as nn
import torch.nn.functional as f
# import MinkowskiEngine as ME
import numpy as np
from torch.nn.modules.activation import ReLU


class InceptionBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.conv(x)

class DilatedBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        cardinatity=2,
        activation="relu",
        norm=None,
        BN_momentum=0.1,):
        super().__init__()
        self.DConv1 = nn.ModuleList([InceptionBlock(in_channels=in_channels, out_channels=out_channels, 
        kernel_size=kernel_size, stride=stride, padding=padding, dilation=1) for _ in range(cardinatity)
        ])
        self.DConv2 = nn.ModuleList([InceptionBlock(in_channels=in_channels, out_channels=out_channels, 
        kernel_size=kernel_size, stride=stride, padding=padding, dilation=2) for _ in range(cardinatity)
        ])
        self.DConv3 = nn.ModuleList([InceptionBlock(in_channels=in_channels, out_channels=out_channels, 
        kernel_size=kernel_size, stride=stride, padding=padding, dilation=3) for _ in range(cardinatity)
        ])

    def forward(self, x):
        out = 0
        
        for body in self.DConv1:
            out += body(x)
        for body in self.DConv2:
            out += body(x)
        for body in self.DConv3:
            out += body(x)
 
        return out
        


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = f.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: torch.tensor, BxNxC
        return: torch.tensor, BxNxC
        """
        x = x.transpose(1, 2) # BxCxN

        x_q = self.q_conv(x).permute(0, 2, 1) # BxNxC1
        x_k = self.k_conv(x) # BxC1xN      
        x_v = self.v_conv(x) # BxCxN
        energy = x_q @ x_k # BxNxN
        attention = self.softmax(energy) # BxNxN
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True)) # BxNxN
        x_r = x_v @ attention # BxCxN
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r))) # BxCxN
        x = x + x_r # BxCxN

        x = x.transpose(1, 2) # BxNxC

        return x


class ConvLayer1D(nn.Module):
    """
    Convolutional layer.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation="relu",
        norm=None,
        BN_momentum=0.1,
    ):
        super(ConvLayer1D, self).__init__()

        bias = False if norm == "BN" else True
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm1d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm1d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv1d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out
        

class ConvLayer(nn.Module):
    """
    Convolutional layer.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation="ReLU",
        norm=None,
        BN_momentum=0.1,
    ):
        super(ConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(nn, activation)()
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class ConvLayer3D(nn.Module):
    """
    Convolutional layer.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation="ReLU",
        norm=None,
        BN_momentum=0.1,
    ):
        super(ConvLayer3D, self).__init__()

        bias = False if norm == "BN" else True
        self.conv2d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(nn, activation)()
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposedConvLayer(nn.Module):
    """
    Transposed convolutional layer to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        activation="relu",
        norm=None,
    ):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=bias,
        )

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    """
    Upsampling layer (bilinear interpolation + Conv2d) to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation="ReLU",
        norm=None,
        scale=2
    ):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            # self.activation = getattr(torch, activation)
            self.activation = getattr(nn, activation)()
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

        self.scale = scale

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    """
    Layer comprised of a convolution followed by a recurrent convolutional block.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        recurrent_block_type="convlstm",
        activation="ReLU",
        norm=None,
        BN_momentum=0.1,
    ):
        super(RecurrentConvLayer, self).__init__()

        assert recurrent_block_type in ["convlstm", "convgru"]
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == "convlstm":
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ConvLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation,
            norm,
            BN_momentum=BN_momentum,
        )
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == "convlstm" else state
        return x, state


class ResidualBlock(nn.Module):
    """
    Residual block as in "Deep residual learning for image recognition", He et al. 2016.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        norm=None,
        BN_momentum=0.1,
        activation="ReLU",
        final_activation=True
    ):
        super(ResidualBlock, self).__init__()
        self.final_activation = final_activation
        bias = False if norm == "BN" else True
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )
        self.norm = norm
        if norm == "BN":
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        # self.activation = getattr(torch, activation)
        self.activation = getattr(nn, activation)()
        # self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ["BN", "IN"]:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm in ["BN", "IN"]:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        if self.final_activation:
            out = self.activation(out)
        return out


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM module.
    Adapted from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvGRU(nn.Module):
    """
    Convolutional GRU cell.
    Adapted from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.0)
        nn.init.constant_(self.update_gate.bias, 0.0)
        nn.init.constant_(self.out_gate.bias, 0.0)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


# submodules for 3D models
def conv_block_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation_type='LeakyReLU'):
    if activation_type == 'ReLU':
        activation = nn.ReLU(inplace=True)
    elif activation_type == 'LeakyReLU':
        activation = nn.LeakyReLU(inplace=True)
    elif activation_type == 'tanh':
        activation = nn.Tanh()
    elif activation_type == None:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels))

    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm3d(out_channels),
        activation,)


def deconv_block_3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, activation_type='LeakyReLU'):
    if activation_type == 'ReLU':
        activation = nn.ReLU(inplace=True)
    elif activation_type == 'LeakyReLU':
        activation = nn.LeakyReLU(inplace=True)
    elif activation_type == 'tanh':
        activation = nn.Tanh()
    elif activation_type == None:
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm3d(out_channels))

    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
        nn.BatchNorm3d(out_channels),
        activation,)


def conv_block_2_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_kernel=2, pool_stride=2, pool_padding=0, activation_type='LeakyReLU'):
    return nn.Sequential(
        conv_block_3d(in_channels, in_channels, kernel_size, stride, padding, activation_type),
        conv_block_3d(in_channels, out_channels, kernel_size, stride, padding, activation_type),
        nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding))


def deconv_block_2_3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, activation_type='LeakyReLU'):
    return nn.Sequential(
        deconv_block_3d(in_channels, out_channels, kernel_size, stride, padding, output_padding, activation_type),
        conv_block_3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation_type='LeakyReLU'),
        conv_block_3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation_type='LeakyReLU'))

# submodules for sparse 3D models
class sparse_resblock(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride, D, dilation=1, bias=False):
        super().__init__()

        self.conv1 = ME.MinkowskiConvolution(inch, outch, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias, dimension=D)
        self.norm1 = ME.MinkowskiBatchNorm(outch)
        self.activation1 = ME.MinkowskiELU()
        self.conv2 = ME.MinkowskiConvolution(outch, outch, kernel_size=3, stride=1, dilation=dilation, bias=bias, dimension=D)
        self.norm2 = ME.MinkowskiBatchNorm(outch)
        self.activation2 = ME.MinkowskiELU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.activation2(out)

        return out


def sparse_conv_block(inch, outch, kernel_size, stride, D, dilation=1, bias=False):
    return nn.Sequential(
        ME.MinkowskiConvolution(inch, outch, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias, dimension=D),
        ME.MinkowskiBatchNorm(outch),
        ME.MinkowskiELU()
    )


def sparse_conv_twoblock(inch, outch, kernel_size, stride, D, dilation=1, bias=False):
    return nn.Sequential(
        ME.MinkowskiConvolution(inch, outch, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias, dimension=D),
        ME.MinkowskiBatchNorm(outch),
        ME.MinkowskiELU(),
        ME.MinkowskiConvolution(outch, outch, kernel_size=3, stride=1, dilation=dilation, bias=bias, dimension=D),
        ME.MinkowskiBatchNorm(outch),
        ME.MinkowskiELU(),
    )


def sparse_trconv_block(inch, outch, kernel_size, stride, D, dilation=1, bias=False):
    return nn.Sequential(
        ME.MinkowskiGenerativeConvolutionTranspose(inch, outch, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias, dimension=D),
        ME.MinkowskiBatchNorm(outch),
        ME.MinkowskiELU(),
        ME.MinkowskiConvolution(outch, outch, kernel_size=3, stride=1, dilation=dilation, bias=bias, dimension=D),
        ME.MinkowskiBatchNorm(outch),
        ME.MinkowskiELU()
    )


#Submodules for space-time zooming lens
def __batch_distance_matrix_general(A, B):
    """
    :param
        A, B [B,N,C], [B,M,C]
    :return
        D [B,N,M]
    """
    r_A = torch.sum(A * A, dim=2, keepdim=True)
    r_B = torch.sum(B * B, dim=2, keepdim=True)
    m = torch.matmul(A, B.permute(0, 2, 1))
    D = r_A - 2 * m + r_B.permute(0, 2, 1)
    return D


def group_knn(k, query, points, unique=True, NCHW=True):
    """
    group batch of points to neighborhoods
    :param
        k: neighborhood size
        query: BxCxM or BxMxC
        points: BxCxN or BxNxC
        unique: neighborhood contains *unique* points
        NCHW: if true, the second dimension is the channel dimension
    :return
        neighbor_points BxCxMxk (if NCHW) or BxMxkxC (otherwise)
        index_batch     BxMxk
        distance_batch  BxMxk
    """
    if NCHW:
        batch_size, channels, num_points = points.size()
        points_trans = points.transpose(2, 1).contiguous()
        query_trans = query.transpose(2, 1).contiguous()
    else:
        points_trans = points.contiguous()
        query_trans = query.contiguous()

    batch_size, num_points, _ = points_trans.size()
    assert(num_points >= k
           ), "points size must be greater or equal to k"

    D = __batch_distance_matrix_general(query_trans, points_trans)
    if unique:
        # prepare duplicate entries
        points_np = points_trans.detach().cpu().numpy()
        indices_duplicated = np.ones(
            (batch_size, 1, num_points), dtype=np.int32)

        for idx in range(batch_size):
            _, indices = np.unique(points_np[idx], return_index=True, axis=0)
            indices_duplicated[idx, :, indices] = 0

        indices_duplicated = torch.from_numpy(
            indices_duplicated).to(device=D.device, dtype=torch.float32)
        D += torch.max(D) * indices_duplicated

    # (B,M,k)
    distances, point_indices = torch.topk(-D, k, dim=-1, sorted=True)
    # (B,N,C)->(B,M,N,C), (B,M,k)->(B,M,k,C)
    knn_trans = torch.gather(points_trans.unsqueeze(1).expand(-1, query_trans.size(1), -1, -1),
                             2,
                             point_indices.unsqueeze(-1).expand(-1, -1, -1, points_trans.size(-1)))

    if NCHW:
        knn_trans = knn_trans.permute(0, 3, 1, 2)

    return knn_trans, point_indices, -distances


class DenseEdgeConv(nn.Module):
    """docstring for EdgeConv"""

    def __init__(self, in_channels, growth_rate, n, k, **kwargs):
        super(DenseEdgeConv, self).__init__()
        self.growth_rate = growth_rate
        self.n = n
        self.k = k
        self.mlps = torch.nn.ModuleList()
        self.mlps.append(torch.nn.Conv2d(
            2 * in_channels, growth_rate, 1, bias=True))
        for i in range(1, n):
            in_channels += growth_rate
            self.mlps.append(torch.nn.Conv2d(
                in_channels, growth_rate, 1, bias=True))

    def get_local_graph(self, x, k, idx=None):
        """Construct edge feature [x, NN_i - x] for each point x
        :param
            x: (B, C, N)
            k: int
            idx: (B, N, k)
        :return
            edge features: (B, C, N, k)
        """
        if idx is None:
            # BCN(K+1), BN(K+1)
            knn_point, idx, _ = group_knn(k + 1, x, x, unique=True)
            idx = idx[:, :, 1:]
            knn_point = knn_point[:, :, :, 1:]

        neighbor_center = torch.unsqueeze(x, dim=-1)
        neighbor_center = neighbor_center.expand_as(knn_point)

        edge_feature = torch.cat(
            [neighbor_center, knn_point - neighbor_center], dim=1)
        return edge_feature, idx

    def forward(self, x, idx=None):
        """
        args:
            x features (B,C,N)
        return:
            y features (B,C',N)
            idx fknn index (B,C,N,K)
        """
        # [B 2C N K]
        for i, mlp in enumerate(self.mlps):
            if i == 0:
                y, idx = self.get_local_graph(x, k=self.k, idx=idx)
                x = x.unsqueeze(-1).repeat(1, 1, 1, self.k)
                y = torch.cat([nn.functional.relu_(mlp(y)), x], dim=1)
            elif i == (self.n - 1):
                y = torch.cat([mlp(y), y], dim=1)
            else:
                y = torch.cat([nn.functional.relu_(mlp(y)), y], dim=1)

        y, _ = torch.max(y, dim=-1)
        return y, idx


#################### for SRFBN
from collections import OrderedDict
import sys


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type =='bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!'%norm_type)
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!'%pad_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
    padding = (kernel_size-1) // 2
    return padding


def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!'%act_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict'%sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,\
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)


def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, \
                act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, deconv)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False



