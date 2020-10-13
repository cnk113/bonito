"""
Bonito Model template
"""

import torch.nn as nn
from torch import sigmoid
from torch.jit import script
from torch.autograd import Function
from torch.nn import ReLU, LeakyReLU
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout, AvgPool1d, AdaptiveAvgPool1d
from torch.nn.functional import log_softmax, interpolate

from fast_ctc_decode import beam_search, viterbi_search


@script
def swish_jit_fwd(x):
    return x * sigmoid(x)


@script
def swish_jit_bwd(x, grad):
    x_s = sigmoid(x)
    return grad * (x_s * (1 + x * (1 - x_s)))


class SwishAutoFn(Function):

    @staticmethod
    def symbolic(g, x):
        return g.op('Swish', x)

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad)


class Swish(Module):
    """
    Swish Activation function

    https://arxiv.org/abs/1710.05941
    """
    def forward(self, x):
        return SwishAutoFn.apply(x)


activations = {
    "relu": ReLU,
    "swish": Swish,
}


class Model(Module):
    """
    Model template for QuartzNet style architectures

    https://arxiv.org/pdf/1910.10261.pdf
    """
    def __init__(self, config):
        super(Model, self).__init__()
        if 'qscore' not in config:
            self.qbias = 0.0
            self.qscale = 1.0
        else:
            self.qbias = config['qscore']['bias']
            self.qscale = config['qscore']['scale']

        self.config = config
        self.stride = config['block'][0]['stride'][0]
        self.alphabet = config['labels']['labels']
        self.features = config['block'][-1]['filters']
        self.encoder = Encoder(config)
        self.decoder = Decoder(self.features, len(self.alphabet))

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        if beamsize == 1 or qscores:
            seq, path  = viterbi_search(x, self.alphabet, qscores, self.qscale, self.qbias)
        else:
            seq, path = beam_search(x, self.alphabet, beamsize, threshold)
        if return_path: return seq, path
        return seq


class Encoder(Module):
    """
    Builds the model encoder
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        features = self.config['input']['features']
        activation = activations[self.config['encoder']['activation']]()
        encoder_layers = []

        for layer in self.config['block']:
            encoder_layers.append(
                Block(
                    features, layer['filters'], activation,
                    repeat=layer['repeat'], kernel_size=layer['kernel'],
                    stride=layer['stride'], dilation=layer['dilation'],
                    dropout=layer['dropout'], residual=layer['residual'],
                    separable=layer['separable'],
                )
            )

            features = layer['filters']

        self.encoder = Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)


class TCSConv1d(Module):
    """
    Time-Channel Separable 1D Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, separable=False):

        super(TCSConv1d, self).__init__()
        self.separable = separable

        if separable:
            self.depthwise = Conv1d(
                in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias, groups=in_channels
            )

            self.pointwise = Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1,
                dilation=dilation, bias=bias, padding=0
            )
        else:
            self.conv = Conv1d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            )

    def forward(self, x):
        if self.separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        return x


class Block(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False):

        super(Block, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()

        _in_channels = in_channels
        padding = self.get_padding(kernel_size[0], stride[0], dilation[0])

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable
                )
            )

            self.conv.extend(self.get_activation(activation, dropout))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable
            )
        )

        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs(in_channels, out_channels))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation(activation, dropout))

    def get_activation(self, activation, dropout):
        return activation, Dropout(p=dropout)

    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        if self.use_res:
            _x += self.residual(x)
        return self.activation(_x)


class Decoder(Module):
    """
    Decoder
    """
    def __init__(self, features, classes):
        super(Decoder, self).__init__()
        self.layers = Sequential(Conv1d(features, classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.layers(x).permute(2, 0, 1)
        return log_softmax(x, dim=2)


class SqueezeExcite(Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        context_window: int = -1,
        interpolation_mode: str = 'nearest',
        activation: Optional[Callable] = None,
    ):
        """
        Squeeze-and-Excitation sub-module.
        Args:
            channels: Input number of channels.
            reduction_ratio: Reduction ratio for "squeeze" layer.
            context_window: Integer number of timesteps that the context
                should be computed over, using stride 1 average pooling.
                If value < 1, then global context is computed.
            interpolation_mode: Interpolation mode of timestep dimension.
                Used only if context window is > 1.
                The modes available for resizing are: `nearest`, `linear` (3D-only),
                `bilinear`, `area`
            activation: Intermediate activation function used. Must be a
                callable activation function.
        """
        super(SqueezeExcite, self).__init__()
        self.context_window = int(context_window)
        self.interpolation_mode = interpolation_mode

        if self.context_window <= 0:
            self.pool = AdaptiveAvgPool1d(1)  # context window = T
        else:
            self.pool = AvgPool1d(self.context_window, stride=1)

        if activation is None:
            activation = ReLU(inplace=True)

        self.fc = nn.Sequential(
            Linear(channels, channels // reduction_ratio, bias=False),
            activation,
            Linear(channels // reduction_ratio, channels, bias=False),
        )

    def forward(self, x):
        # The use of negative indices on the transpose allow for expanded SqueezeExcite
        batch, channels, timesteps = x.size()[:3]
        y = self.pool(x)  # [B, C, T - context_window + 1]
        y = y.transpose(1, -1)  # [B, T - context_window + 1, C]
        y = self.fc(y)  # [B, T - context_window + 1, C]
        y = y.transpose(1, -1)  # [B, C, T - context_window + 1]

        if self.context_window > 0:
            y = interpolate(y, size=timesteps, mode=self.interpolation_mode)

        y = sigmoid(y)

        return x * y
