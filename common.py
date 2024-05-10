from typing import NoReturn
import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.nn.init as init
import math
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from einops import rearrange
from torch.nn.modules.batchnorm import _BatchNorm
import time
import sys
from loss import optical_flow_warp
import warnings
import collections.abc
from itertools import repeat


###########################################################################
#////////////////////////////default_conv/////////////////////////////////#
###########################################################################
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)



###########################################################################
#////////////////////////////Upsampler////////////////////////////////////#
###########################################################################
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feat, 4 * n_feat, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if bn: modules.append(nn.BatchNorm2d(n_feat))
                if act: modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feat, 9 * n_feat, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if bn: modules.append(nn.BatchNorm2d(n_feat))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)


###########################################################################
#//////////////////////////initialize_weights/////////////////////////////#
###########################################################################
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


###########################################################################
#////////////////Res_Block&Res_Block_s&Conv_ReLU_Block////////////////////#
###########################################################################

class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Localconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Localconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=1, padding = 1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=2, padding = 2, bias=bias)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=3, padding = 3, bias=bias)
        self.cf = default_conv(in_channelss = 3*out_channels, out_channels = out_channels, kernel_size=1)

    def forward(self, x, local):
        if local == 2:
            out1 = self.conv1(x)
            out2 = self.conv2(x)
            out3 = self.conv3(x)
            out = self.cf(torch.cat([out1, out2, out3], dim=1))
        elif local == 1:
            out = self.conv1(x)
        return out
        

class SEBasicBlock(nn.Module):
    def __init__(self):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.cca = CALayer(channel = 64) 

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.cca(res)
        return x + res


class SA_RCAB(nn.Module):
    def __init__(self, conv, nf, kernel_size, reduction, bias=True):
        super(SA_RCAB, self).__init__()

        self.conv1 = conv(nf, nf, kernel_size, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(nf, nf, kernel_size, bias=bias)
        self.SA_exconv = SA_exconv(channels_in=nf, channels_out=nf)
        self.rcab = RCAB(conv, nf, kernel_size, reduction)

    def forward(self, x, scale_w, scale_h):

        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.SA_exconv(res, scale_w, scale_h)
        res = self.rcab(res)
        res = res + x
        return res



###########################################################################
#//////////////////////////Resnet_Dense_Block/////////////////////////////#
###########################################################################
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return torch.cat([x, res],1)


###########################################################################
#///////////////////////////////RDN附属函数////////////////////////////////#
###########################################################################
class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


###########################################################################
#///////////////////////////////Mult_RCAN/////////////////////////////////#
###########################################################################
class Multisconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Multisconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=1, padding = 1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=2, padding = 2, bias=bias)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=3, padding = 3, bias=bias)
        self.cf = default_conv(in_channelss = 3*in_channels, out_channels = out_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = self.cf(torch.cat([out1, out2, out3], dim=1))
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                default_conv, n_feat, kernel_size, reduction=16, act=nn.ReLU(True)) \
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)



###########################################################################
#/////////////////////////LightMLPInterpolate/////////////////////////////#
###########################################################################
def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class LightMLPInterpolate(nn.Module):
    def __init__(self, n_feat, radius=3):
        super().__init__()
        self.radius = radius

        self.out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feat, 3, kernel_size=1, padding=0),
        )

        self.weight_calc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(True),
            nn.Linear(64, self.radius ** 2),
            nn.Softmax(2),
        )

    def forward(self, x, out_size):
        # scale-aware fusion weight calculation
        if type(out_size) == int:
            out_size = [out_size, out_size]

        _device = x.device
        _scale = torch.tensor([x.shape[2] / out_size[0]], device=_device)

        in_shape = x.shape[-2:]
        in_coord = (
            make_coord(in_shape, flatten=False)
            .to(x.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(x.shape[0], 2, *in_shape)
        )

        out_coord = make_coord(out_size, flatten=True).to(_device)
        out_coord = out_coord.expand(x.shape[0], *out_coord.shape)

        q_coord = F.grid_sample(
            in_coord,
            out_coord.flip(-1).unsqueeze(1),
            mode="nearest",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        rel_coord = out_coord - q_coord
        rel_coord[:, :, 0] *= x.shape[-2]
        rel_coord[:, :, 1] *= x.shape[-1]

        scale_tensor = _scale.expand(rel_coord.shape[:2]).unsqueeze(2)
        inp = torch.cat([rel_coord, scale_tensor], dim=2)

        weights = self.weight_calc(inp)

        # feature
        x_unfold = F.unfold(x, self.radius, padding=self.radius // 2)
        x_unfold = x_unfold.view(
            x.shape[0], x.shape[1] * (self.radius ** 2), x.shape[2], x.shape[3]
        )

        q_feat = F.grid_sample(
            x_unfold,
            out_coord.flip(-1).unsqueeze(1),
            mode="nearest",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)

        # fusion
        bs, q = q_feat.shape[:2]
        q_feat = q_feat.view(bs, q, -1, self.radius ** 2)
        out = (q_feat @ weights.unsqueeze(3)).squeeze(3)
        out = rearrange(out, "bs (h w) nf -> bs nf h w", h=out_size[0])
        out = self.out(out)

        return out


###########################################################################
#/////////////////////////////////SA_conv/////////////////////////////////#
###########################################################################
class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att


class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(SA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        out = self.conv(out.view(b, -1, h, w))

        # branch 2
        out = out + self.ca(x)

        return out


###########################################################################
#/////////////////////////////////SA_exconv///////////////////////////////#
###########################################################################
class SA_exconv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=3):
        super(SA_exconv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias
        self.sum = 0

        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0).data)

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale1, scale2):
        # generate routing weights
        scale1 = torch.ones(1, 1).cuda() / scale1.float().cuda()
        scale2 = torch.ones(1, 1).cuda() / scale2.float().cuda()
        scale3 = scale1 / scale2
        routing_weights = self.routing(Variable(torch.cat((scale1, scale2, scale3), 1))).view(self.num_experts, 1, 1)

        # fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.channels_in, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        # convolution
        out = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)

        return out


###########################################################################
#/////////////////////////////////SA_adapt////////////////////////////////#
###########################################################################
class SA_adapt(nn.Module):
    def __init__(self, channels):
        super(SA_adapt, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_exconv(channels, channels, 3, 1, 1)

    def forward(self, x, scale, scale2):
        mask = self.mask(x)
        adapted = self.adapt(x, scale, scale2)

        return x + adapted * mask


###########################################################################
#//////////////////////////////PredeblurModule////////////////////////////#
###########################################################################
class PredeblurModule(nn.Module):
    """Pre-dublur module.
    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super(PredeblurModule, self).__init__()
        self.hr_in = hr_in

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        if self.hr_in:
            # downsample x4 by stride conv
            self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
            self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        # generate feature pyramid
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.ModuleList(
            [ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat_l1 = self.lrelu(self.conv_first(x))
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))

        # generate feature pyramid
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))

        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))

        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)
        feat_l1 = feat_l1 + feat_l2
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)
        return feat_l1


class MYPredeblurModule(nn.Module):
    """Pre-dublur module.
    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super(MYPredeblurModule, self).__init__()
        self.hr_in = hr_in

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        if self.hr_in:
            # downsample x4 by stride conv
            self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
            self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        # generate feature for different frames
        self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 5, 1, 2)
        self.stride_conv_hr3 = nn.Conv2d(num_feat, num_feat, 7, 1, 3)

        self.resblock_l1 = nn.ModuleList(
            [ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, t):

        feat_l1 = self.lrelu(self.conv_first(x))
        b, c, h, w = feat_l1.size()
        b = int(b/t)
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))

        # generate feature pyramid
        feat_l1s = []
        feat_l1 = feat_l1.view(b, t, c, h, w)

        for i in range(t):
            if i==2:
                feat = self.resblock_l1[i](self.stride_conv_hr1(feat_l1[:,i,:,:,:]))
                feat_l1s.append(feat)
            elif i==1 or i==3:
                feat = self.resblock_l1[i](self.stride_conv_hr2(feat_l1[:,i,:,:,:]))
                feat_l1s.append(feat)
            else:
                feat = self.resblock_l1[i](self.stride_conv_hr3(feat_l1[:,i,:,:,:]))
                feat_l1s.append(feat)
        feat_l1 = torch.cat(feat_l1s, dim=1).view(-1, c, h, w)

        return feat_l1

###########################################################################
#////////////////////////////////////TSAFusion////////////////////////////#
###########################################################################
class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.
    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)
    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear')

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).
        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        # for pytorch 0.3.1
        #nn.init.constant(self.W.weight, 0)
        #nn.init.constant(self.W.bias, 0)
        # for pytorch 0.4.0
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        
        g_x = g_x.permute(0,2,1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        
        theta_x = theta_x.permute(0,2,1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
       
        f_div_C = F.softmax(f, dim=1)
        
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0,2,1).contiguous()
         
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NLResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(NLResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class NLMaskBranchDownUp(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(NLMaskBranchDownUp, self).__init__()
        
        MB_RB1 = []
        MB_RB1.append(NonLocalBlock2D(n_feat, n_feat // 2))
        MB_RB1.append(NLResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat,n_feat, 3, stride=2, padding=1))
        
        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(NLResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
         
        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat,n_feat, 6, stride=2, padding=2))   
        
        MB_RB3 = []
        MB_RB3.append(NLResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat,n_feat, 1, padding=0, bias=True))
        
        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up  = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)
    
    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            *[ 
                nn.Conv2d(in_planes, out_planes, kernel_size, padding=(kernel_size-1) // 2, stride=1, bias=False),
                nn.BatchNorm2d(out_planes ,eps=1e-5, momentum=0.01, affine=True)
            ]
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class NMFFM(nn.Module):
    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(NMFFM, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        self.mask = NLMaskBranchDownUp(default_conv, num_feat, 3)    
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).
        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        mask = self.mask(feat)
        feat = feat * mask

        return feat
    


class MYTSAFusion(nn.Module):
    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(MYTSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        self.mask = NLMaskBranchDownUp(default_conv, num_feat, 3)    
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).
        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        mask = self.mask(feat)
        feat = feat * mask

        return feat

###########################################################################
#////////////////////////////////make_layer///////////////////////////////#
###########################################################################
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

###########################################################################
#///////////////////////////CSVSR////////////////////////#
###########################################################################
class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class SARB(nn.Module):

    def __init__(self, num_feat=64, reduction=16, num_experts=3, expend=4, res_scale=1, pytorch_init=False):
        super(SARB, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat*expend, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat*expend, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.SA_exconv = SA_exconv(channels_in=num_feat, channels_out=num_feat, num_experts=num_experts)
        self.rcab = RCAB(default_conv, num_feat, 3, reduction)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x, scale_w, scale_h):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        out = self.SA_exconv(out, scale_w, scale_h)
        out = self.rcab(out)
        return identity + out * self.res_scale
    

class SFEM(nn.Module):
    def __init__(self, num_feat=64, reduction=16, num_block=10, num_experts=3):
        super(SFEM, self).__init__()

        self.SARBs = nn.ModuleList()
        for i in range(num_block):
            self.SARBs.append(SARB(num_feat, reduction, num_experts))
 

    def forward(self, x, scale_w, scale_h):
        out = x
        for SARB in self.SARBs:
            out = SARB(out, scale_w, scale_h)
        return out
    


class MYResidualBlockNoBN(nn.Module):

    def __init__(self, num_feat=64, reduction=16, num_experts=3, expend=4, res_scale=1, pytorch_init=False):
        super(MYResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat*expend, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat*expend, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.SA_exconv = SA_exconv(channels_in=num_feat, channels_out=num_feat, num_experts=num_experts)
        self.rcab = RCAB(default_conv, num_feat, 3, reduction)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x, scale_w, scale_h):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        out = self.SA_exconv(out, scale_w, scale_h)
        out = self.rcab(out)
        return identity + out * self.res_scale
    

class MYResidualBlockNoBNMould(nn.Module):
    def __init__(self, num_feat=64, reduction=16, num_block=10, num_experts=3):
        super(MYResidualBlockNoBNMould, self).__init__()

        self.MYResidualBlockNoBNs = nn.ModuleList()
        for i in range(num_block):
            self.MYResidualBlockNoBNs.append(MYResidualBlockNoBN(num_feat, reduction, num_experts))
 

    def forward(self, x, scale_w, scale_h):
        out = x
        for MYResidualBlockNoBNs in self.MYResidualBlockNoBNs:
            out = MYResidualBlockNoBNs(out, scale_w, scale_h)
        return out


class Offsetconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, pytorch_init=False):
        super(Offsetconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=1, padding = 1, bias=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=2, padding = 2, bias=True)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=3, padding = 3, bias=True)
        self.cat = default_conv(in_channelss = 3*in_channels, out_channels = out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding = 1, bias=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = self.relu(self.cat(torch.cat([out1, out2, out3], dim=1)))
        out = self.conv(out)
        return identity + out

###########################################################################
#////////////////////////////default_init_weights/////////////////////////#
###########################################################################
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.
    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
