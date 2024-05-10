import os
from os.path import exists
from audioop import bias
from re import I
from numpy.core.numeric import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable, Function
import math
import numpy as np
from modules import ConvOffset2d
from common import *
import time
import skimage
from loss import optical_flow_warp
from math import sqrt
import functools
from matplotlib import pyplot as plt


class ModelFactory(object):

    def create_model(self, model_name, scale):
        if len(scale) == 1:
            scale = int(scale[0])
        else:
            scale = 4
        if model_name == "EDVR":
            return EDVR(scale = scale)
        elif model_name == "TDAN":
            return TDAN(scale = scale) 
        elif model_name == "CSVSR":
            return CSVSR(scale = scale)
        elif model_name == "MYEDVR":
            return MYEDVR_VSR(scale = scale) #
        else:
            raise Exception('unknown model {}'.format(model_name))

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_align(self, weights=''):
        net_align = align_net()
        #net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_align')
            #net_align.load_state_dict(torch.load(weights))
            net_align = torch.load(weights)

            if isinstance(net_align, torch.nn.DataParallel):
                net_align = net_align.module
        return net_align

    def build_align_feat(self, weights=''):
        net_align = align_net_w_feat()
        #net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_align')
            net = torch.load(weights)
            model_dict = net_align.state_dict()
            net_dict = net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in net_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net_align.load_state_dict(model_dict)

            if isinstance(net_align, torch.nn.DataParallel):
                net_align = net_align.module
        return net_align

    # builder for vision
    def build_rec(self, num_block=10, weights='', scale=1.0):
        net_rec = SR_Rec(num_block, scale)

        if len(weights) > 0:
            print('Loading weights for net_rec')
            #net_rec.load_state_dict(torch.load(weights))
            net = torch.load(weights)

            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            model_dict = net_rec.state_dict()
            net_dict = net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in net_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net_rec.load_state_dict(model_dict)
        return net_rec

    # builder for vision
    def build_vsr_rec(self, num_block=10, weights='', scale=1.0):
        net_rec = VSR_Rec(num_block, scale)

        if len(weights) > 0:
            print('Loading weights for net_rec')
            #net_rec.load_state_dict(torch.load(weights))
            net = torch.load(weights)

            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            model_dict = net_rec.state_dict()
            net_dict = net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in net_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net_rec.load_state_dict(model_dict)
        return net_rec

class BICUBIC(nn.Module):
    def __init__(self):
        super(BICUBIC, self).__init__()
        self.name = 'BICUBIC'

    def forward(self, x, scale):

        if isinstance(scale, float):
            scale_w = scale
            scale_h = scale
        else:
            scale_w, scale_h = scale
            
        batch_size, num, ch, w, h = x.size()  # 5 video frames

        w_size = int(w*scale_w)
        h_size = int(h*scale_h)
        # w_size = 512
        # h_size = 512
        center = num // 2
        x_center = x[:, center, :, :, :]

        out = skimage.transform.resize(x_center, (w_size, h_size), order=3)

        return out

class TDAN_L(nn.Module):
    def __init__(self, nets):
        super(TDAN_L, self).__init__()
        self.name = 'TDAN_L' #tdan_L with 80 blocks
        self.align_net, self.rec_net = nets

        for param in self.align_net.parameters():
            param.requires_grad = True

    def forward(self, x):

        lrs = self.align_net(x)
        y = self.rec_net(lrs)

        return y, lrs

class TDAN_F(nn.Module):
    def __init__(self, nets):
        super(TDAN_F, self).__init__()
        self.name = 'TDAN_F' #tdan_L with 80 blocks
        self.align_net, self.rec_net = nets

        for param in self.align_net.parameters():
            param.requires_grad = True

    def forward(self, x):

        lrs, feat = self.align_net(x)
        y = self.rec_net(lrs, feat)

        return y, lrs


class TDAN_VSR(nn.Module):
    def __init__(self, scale = 4):
        super(TDAN_VSR, self).__init__()
        self.name = 'TDAN'
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)

        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        # deformable
        self.cr = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_2 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_2 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_3 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv = ConvOffset2d(64, 64, (3, 3), padding=(1, 1), num_deformable_groups=8)
        self.recon_lr = nn.Conv2d(64, 3, 3, padding=1, bias=True)

        fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding= 1, bias=True),
                       nn.ReLU()]

        self.fea_ex = nn.Sequential(*fea_ex)
        self.recon_layer = self.make_layer(Res_Block, 10)
        upscaling = [
            Upsampler(default_conv, scale, 64, act=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]

        self.up = nn.Sequential(*upscaling)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def align(self, x, x_center):
        y = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, center, :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                continue
            supp = x[:, i, :, :, :]
            fea = torch.cat([ref, supp], dim=1)
            fea = self.cr(fea)
            # feature trans
            offset1 = self.off2d_1(fea)
            fea = (self.dconv_1(fea, offset1))
            offset2 = self.off2d_2(fea)
            fea = (self.deconv_2(fea, offset2))
            offset3 = self.off2d_3(fea)
            fea = (self.deconv_3(supp, offset3))
            offset4 = self.off2d(fea)
            aligned_fea = (self.dconv(fea, offset4))
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
        y = torch.cat(y, dim=1)
        return y

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, scale):

        batch_size, num, ch, w, h = x.size()  # 5 video frames
        # center frame interpolation
        center = num // 2
        # extract features
        y = x.view(-1, ch, w, h)
        # y = y.unsqueeze(1)
        out = self.relu(self.conv_first(y))
        x_center = x[:, center, :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)

        # align supporting frames
        lrs = self.align(out, x_center) # motion alignments
        y = lrs.view(batch_size, -1, w, h)
        # reconstruction
        fea = self.fea_ex(y)

        out = self.recon_layer(fea)
        out = self.up(out)
        if scale!=4:
            if isinstance(scale, list):
                h_size = int(w*scale[0])
                w_size = int(h*scale[1])
            else:
                h_size = int(w*scale)
                w_size = int(h*scale)
            out = nn.functional.upsample(out, size=(h_size, w_size), mode='bilinear', align_corners=False)
        return out, lrs

# vsr network
class TDAN(nn.Module):
    def __init__(self, nets):
        super(TDAN, self).__init__()
        self.name = 'TDAN'
        self.align_net, self.rec_net = nets

    def forward(self, x):

        lrs = self.align_net(x)
        y = self.rec_net(lrs)

        return y, lrs


# alignment network
class align_net_w_feat(nn.Module):
    def __init__(self):
        super(align_net_w_feat, self).__init__()

        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)

        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)

        # deformable
        self.cr = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_2 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_2 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_3 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv = ConvOffset2d(64, 64, (3, 3), padding=(1, 1), num_deformable_groups=8)
        self.recon_lr = nn.Conv2d(64, 3, 3, padding=1, bias=True)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def align(self, x, x_center):
        y = []
        feats = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, center, :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                feats.append(ref.unsqueeze(1))
                continue
            supp = x[:, i, :, :, :]
            fea = torch.cat([ref, supp], dim=1)
            fea = self.cr(fea)

            # feature trans
            offset1 = self.off2d_1(fea)
            fea = self.dconv_1(fea, offset1)
            offset2 = self.off2d_2(fea)
            fea = self.deconv_2(fea, offset2)
            offset3 = self.off2d_3(fea)
            fea = self.deconv_3(supp, offset3)
            offset4 = self.off2d(fea)
            aligned_fea = self.dconv(fea, offset4)
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
            feats.append(fea.unsqueeze(1))
        y = torch.cat(y, dim=1)
        feats = torch.cat(feats, dim=1)

        return y, feats

    def forward(self, x):
        batch_size, num, ch, w, h = x.size()  # 5 video frames

        # center frame interpolation
        center = num // 2

        # extract features
        y = x.view(-1, ch, w, h)
        out = self.relu(self.conv_first(y))
        x_center = x[:, center, :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)

        # align supporting frames
        lrs, feats = self.align(out, x_center)  # motion alignments
        return lrs, feats


class align_net(nn.Module):
    def __init__(self):
        super(align_net, self).__init__()

        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)

        # deformable
        self.cr = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_2 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_2 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_3 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv = ConvOffset2d(64, 64, (3, 3), padding=(1, 1), num_deformable_groups=8)
        self.recon_lr = nn.Conv2d(64, 3, 3, padding=1, bias=True)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def align(self, x, x_center):
        y = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, center, :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                continue
            supp = x[:, i, :, :, :]
            fea = torch.cat([ref, supp], dim=1)
            fea = self.cr(fea)

            # feature trans
            offset1 = self.off2d_1(fea)
            fea = self.dconv_1(fea, offset1)
            offset2 = self.off2d_2(fea)
            fea = self.deconv_2(fea, offset2)
            offset3 = self.off2d_3(fea)
            fea = self.deconv_3(supp, offset3)
            offset4 = self.off2d(fea)
            aligned_fea = self.dconv(fea, offset4)
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
        y = torch.cat(y, dim=1)
        return y

    def forward(self, x):
        batch_size, num, ch, w, h = x.size()  # 5 video frames

        # center frame interpolation
        center = num // 2

        # extract features
        y = x.view(-1, ch, w, h)
        out = self.relu(self.conv_first(y))
        x_center = x[:, center, :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)
        # align supporting frames
        lrs = self.align(out, x_center)  # motion alignments
        return lrs


# sr reconstruction network
class SR_Rec(nn.Module):
    def __init__(self, nb_block=10, scale=1.0):
        super(SR_Rec, self).__init__()
        self.recon_layer = self.make_layer(Res_Block_s(scale), nb_block)
        fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding=1, bias=True),
                  nn.ReLU()]
        self.fea_ex = nn.Sequential(*fea_ex)
        upscaling = [
            Upsampler(default_conv, 4, 64, act=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]
        self.up = nn.Sequential(*upscaling)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, y):
        batch_size, num, ch, w, h = y.size()
        center = num // 2
        #y_center = y[:, center, :, :, :]
        y = y.view(batch_size, -1, w, h)
        fea = self.fea_ex(y)
        out = self.recon_layer(fea)
        out = self.up(out)
        return out #+ F.upsample(y_center, scale_factor=4, mode='bilinear')

class VSR_Rec(nn.Module):
    def __init__(self, nb_block=10, scale=1.0):
        super(VSR_Rec, self).__init__()

        fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding=1, bias=True),
                  nn.ReLU()]
        self.fea_ex = nn.Sequential(*fea_ex)
        self.fuse = nn.Conv2d(6*64, 64, 3, padding=1, bias=True)

        self.recon_layer = self.make_layer(Res_Block_s(scale), nb_block)
        upscaling = [
            Upsampler(default_conv, 4, 64, act=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]
        self.up = nn.Sequential(*upscaling)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, y, feats):
        batch_size, num, ch, w, h = y.size()
        center = num // 2
        #y_center = y[:, center, :, :, :]
        y = y.view(batch_size, -1, w, h)
        feat = self.fea_ex(y)
        feat = torch.cat((feats, feat.unsqueeze(1)), 1).view(batch_size, -1, w, h)
        feat = self.fuse(feat)
        out = self.recon_layer(feat)
        out = self.up(out)
        return out
      
      

class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.
    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleList()
        self.offset_conv2 = nn.ModuleList()
        self.offset_conv3 = nn.ModuleList()
        self.dcn_pack = nn.ModuleList()
        self.feat_conv = nn.ModuleList()

        # Pyramids
        for i in range(3, 0, -1):
            level = i-1
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, 18 * 8, 3, 1,
                                                 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(18 * 8, 18 * 8, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = nn.Conv2d(18 * 8 * 2, 18 * 8, 3,
                                                     1, 1)
                self.offset_conv3[level] = nn.Conv2d(18 * 8, 18 * 8, 3, 1,
                                                     1)
            self.dcn_pack[level] = ConvOffset2d(
                num_feat,
                num_feat,
                3,
                padding=1,
                num_deformable_groups = deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, 18 * 8, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(18 * 8, 18 * 8, 3, 1, 1)
        self.cas_dcnpack = ConvOffset2d(
            num_feat,
            num_feat,
            3,
            padding=1,
            num_deformable_groups = deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.
        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = i-1
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat

class EDVR(nn.Module):
    """EDVR network structure for video super-resolution.
    Now only support X4 upsampling factor.
    Paper:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(self,
                 scale,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=2,
                 hr_in=False,
                 with_predeblur=False,
                 with_tsa=True):
        super(EDVR, self).__init__()
        self.name = 'EDVR'
        self.scale = scale
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa

        # extract features for each frame
        if self.with_predeblur:
            self.predeblur = PredeblurModule(
                num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extrat pyramid features
        self.feature_extraction = make_layer(
            ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(
            num_feat=num_feat, deformable_groups=deformable_groups)
        if self.with_tsa:
            self.fusion = TSAFusion(
                num_feat=num_feat,
                num_frame=num_frame,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = make_layer(
            ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, scale):
        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, (
                'The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, (
                'The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        if self.hr_in:
            base = x_center
        else:
            base = nn.functional.upsample(x_center, scale_factor=int(4), mode='bilinear')
        out += base
        if scale!=4:
            if isinstance(scale, list):
                h_size = int(h*scale[0])
                w_size = int(w*scale[1])
            else:
                h_size = int(h*scale)
                w_size = int(w*scale)
            out = nn.functional.upsample(out, size=(h_size, w_size), mode='bilinear', align_corners=False)
        return out


class RTDRAM(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8):
        super(RTDRAM, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleList()
        self.offset_conv2 = nn.ModuleList()
        self.offset_conv3 = nn.ModuleList()
        self.dcn_pack = nn.ModuleList()
        self.feat_conv = nn.ModuleList()

        # Pyramids
        for i in range(3, 0, -1):
            level = i-1
            self.offset_conv1[level] = nn.Conv2d(num_feat * 3, 18 * 8, 3, 1,
                                                 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(18 * 8, 18 * 8, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = nn.Conv2d(18 * 8 * 2, 18 * 8, 3,
                                                     1, 1)
                self.offset_conv3[level] = nn.Conv2d(18 * 8, 18 * 8, 3, 1,
                                                     1)
            self.dcn_pack[level] = ConvOffset2d(
                num_feat,
                num_feat,
                3,
                padding=1,
                num_deformable_groups = deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, 18 * 8, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(18 * 8, 18 * 8, 3, 1, 1)
        self.cas_offset_conv3 = Offsetconv(18 * 8, 18 * 8, 3)
        self.cas_dcnpack = ConvOffset2d(
            num_feat,
            num_feat,
            3,
            padding=1,
            num_deformable_groups = deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l, local):
        """Align neighboring frame features to the reference frame features.
        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = i-1
            flowminus = ref_feat_l[i - 1] - nbr_feat_l[i - 1]

            offset = torch.cat([nbr_feat_l[i - 1]*torch.sigmoid(flowminus), ref_feat_l[i - 1]*torch.sigmoid(flowminus), flowminus], dim=1)
            # offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        if abs(local-2)==1 or abs(local-2)==0:
            offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        else:
            offset = self.lrelu(self.cas_offset_conv3(self.lrelu(self.cas_offset_conv1(offset))))

        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat
    

class MYPCDAlignment(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8):
        super(MYPCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleList()
        self.offset_conv2 = nn.ModuleList()
        self.offset_conv3 = nn.ModuleList()
        self.dcn_pack = nn.ModuleList()
        self.feat_conv = nn.ModuleList()

        # Pyramids
        for i in range(3, 0, -1):
            level = i-1
            self.offset_conv1[level] = nn.Conv2d(num_feat * 3, 18 * 8, 3, 1,
                                                 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(18 * 8, 18 * 8, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = nn.Conv2d(18 * 8 * 2, 18 * 8, 3,
                                                     1, 1)
                self.offset_conv3[level] = nn.Conv2d(18 * 8, 18 * 8, 3, 1,
                                                     1)
            self.dcn_pack[level] = ConvOffset2d(
                num_feat,
                num_feat,
                3,
                padding=1,
                num_deformable_groups = deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, 18 * 8, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(18 * 8, 18 * 8, 3, 1, 1)
        self.cas_offset_conv3 = Offsetconv(18 * 8, 18 * 8, 3)
        self.cas_dcnpack = ConvOffset2d(
            num_feat,
            num_feat,
            3,
            padding=1,
            num_deformable_groups = deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l, local):
        """Align neighboring frame features to the reference frame features.
        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = i-1
            flowminus = ref_feat_l[i - 1] - nbr_feat_l[i - 1]

            offset = torch.cat([nbr_feat_l[i - 1]*torch.sigmoid(flowminus), ref_feat_l[i - 1]*torch.sigmoid(flowminus), flowminus], dim=1)
            # offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        if abs(local-2)==1 or abs(local-2)==0:
            offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        else:
            offset = self.lrelu(self.cas_offset_conv3(self.lrelu(self.cas_offset_conv1(offset))))

        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat



class CSVSR(nn.Module):
    """CSVSR network structure for video super-resolution.
    """
    def __init__(self,
                 scale,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=6,
                 num_reconstruct_block=12,
                 center_frame_idx=2,
                 num_experts=3,
                 hr_in=False,
                 with_predeblur=False,
                 with_tsa=True):
        super(CSVSR, self).__init__()
        self.name = 'CSVSR'
        self.scale = scale
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa

        # extract features for each frame
        if self.with_predeblur:
            self.predeblur = PredeblurModule(
                num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extrat pyramid features
        self.feature_extraction = make_layer(
            ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # RTDRAM and NMFFM module
        self.pcd_align = RTDRAM(
            num_feat=num_feat, deformable_groups=deformable_groups)
        if self.with_tsa:
            self.fusion = NMFFM(
                num_feat=num_feat,
                num_frame=num_frame,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # SFEM
        
        self.reconstruction = SFEM(num_feat=64, reduction=16, num_block=num_reconstruct_block, num_experts = num_experts)

        # MPACUM
        self.upconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.att1 = PA(num_feat)
        self.HRconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        
        self.upconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.att2 = PA(num_feat)
        self.HRconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            
        # self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1, bias=True)
        self.conv_last = nn.Sequential(
            *[ 
                nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_feat, 3, kernel_size=1, padding=0, bias=False)
            ]
        )
        self.lreluup = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fuse_scale = nn.Conv2d(3 * num_feat, num_feat, 1, 1)

    def forward(self, x, scale):
        
        if isinstance(scale, torch.Tensor):
            scale_h = scale
            scale_w = scale  
        else:
            scale_h, scale_w = scale
            
        b, t, c, h, w = x.size()

        h_size = int(h*scale_h)
        w_size = int(w*scale_w)

        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, (
                'The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, (
                'The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w),t))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l, i))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.reconstruction(feat, scale_h, scale_w)
        fea1 = self.upconv1(nn.functional.upsample(out, scale_factor=2, mode='nearest'))
        fea1 = self.lreluup(self.att1(fea1))
        fea1 = self.lreluup(self.HRconv1(fea1))
        fea2 = self.upconv2(nn.functional.upsample(fea1, scale_factor=2, mode='nearest'))
        fea2 = self.lreluup(self.att2(fea2))
        fea2 = self.lreluup(self.HRconv2(fea2))
        fea1 = nn.functional.upsample(fea1, size=(h_size, w_size), mode='bilinear')
        fea2 = nn.functional.upsample(fea2, size=(h_size, w_size), mode='bilinear')
        fea3 = nn.functional.upsample(out, size=(h_size, w_size), mode='bilinear')
        fea = self.fuse_scale(torch.cat([fea1, fea2, fea3], dim=1))
        
        out = self.conv_last(fea)

        if self.hr_in:
            base = x_center
        else:
            base = nn.functional.upsample(x_center, size=(h_size, w_size), mode='bilinear')
        out += base
        return out



class MYEDVR_VSR(nn.Module):
    """CSVSR network structure for video super-resolution.
    """
    def __init__(self,
                 scale,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=6,
                 num_reconstruct_block=12,
                 center_frame_idx=2,
                 num_experts=3,
                 hr_in=False,
                 with_predeblur=False,
                 with_tsa=True):
        super(MYEDVR_VSR, self).__init__()
        self.name = 'CSVSR'
        self.scale = scale
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa

        # extract features for each frame
        if self.with_predeblur:
            self.predeblur = PredeblurModule(
                num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extrat pyramid features
        self.feature_extraction = make_layer(
            ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # RTDRAM and NMFFM module
        self.pcd_align = MYPCDAlignment(
            num_feat=num_feat, deformable_groups=deformable_groups)
        if self.with_tsa:
            self.fusion = MYTSAFusion(
                num_feat=num_feat,
                num_frame=num_frame,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # SFEM
        self.reconstruction = MYResidualBlockNoBNMould(num_feat=64, reduction=16, num_block=num_reconstruct_block, num_experts = num_experts)
        # MPACUM
        self.upconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.att1 = PA(num_feat)
        self.HRconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        
        self.upconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.att2 = PA(num_feat)
        self.HRconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            
        # self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1, bias=True)
        self.conv_last = nn.Sequential(
            *[ 
                nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_feat, 3, kernel_size=1, padding=0, bias=False)
            ]
        )
        self.lreluup = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fuse_scale = nn.Conv2d(3 * num_feat, num_feat, 1, 1)

    def forward(self, x, scale):
        
        if isinstance(scale, torch.Tensor):
            scale_h = scale
            scale_w = scale  
        else:
            scale_h, scale_w = scale
            
        b, t, c, h, w = x.size()

        h_size = int(h*scale_h)
        w_size = int(w*scale_w)

        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, (
                'The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, (
                'The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w),t))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l, i))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.reconstruction(feat, scale_h, scale_w)
        # out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        # out = self.lrelu(self.conv_hr(out))
        # out = self.conv_last(out)
        #=======================================================
        fea1 = self.upconv1(nn.functional.upsample(out, scale_factor=2, mode='nearest'))
        fea1 = self.lreluup(self.att1(fea1))
        fea1 = self.lreluup(self.HRconv1(fea1))
        fea2 = self.upconv2(nn.functional.upsample(fea1, scale_factor=2, mode='nearest'))
        fea2 = self.lreluup(self.att2(fea2))
        fea2 = self.lreluup(self.HRconv2(fea2))
        fea1 = nn.functional.upsample(fea1, size=(h_size, w_size), mode='bilinear')
        fea2 = nn.functional.upsample(fea2, size=(h_size, w_size), mode='bilinear')
        fea3 = nn.functional.upsample(out, size=(h_size, w_size), mode='bilinear')
        fea = self.fuse_scale(torch.cat([fea1, fea2, fea3], dim=1))
        
        out = self.conv_last(fea)

        if self.hr_in:
            base = x_center
        else:
            base = nn.functional.upsample(x_center, size=(h_size, w_size), mode='bilinear')
        out += base
        return out