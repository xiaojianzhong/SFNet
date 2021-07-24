import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (build_conv_layer, ConvModule)
from mmseg.models.builder import NECKS
from mmseg.models.decode_heads.psp_head import PPM
from mmcv.runner import auto_fp16


class FAM(nn.Module):
    """Flow Alignment Module used in SFNet.

    Args:
        low_in_channels (int): Input channels of low-level features.
        high_in_channels (int): Input channels of high-level features.
        out_channels (int): Output channels.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
    """

    def __init__(self,
                 low_in_channels,
                 high_in_channels,
                 out_channels,
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(FAM, self).__init__()
        self.low_in_channels = low_in_channels
        self.high_in_channels = high_in_channels
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.low_conv = ConvModule(
            self.low_in_channels,
            self.high_in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.low_down_conv = build_conv_layer(
            self.conv_cfg,
            self.high_in_channels,
            self.out_channels,
            kernel_size=1,
            bias=False)
        self.high_down_conv = build_conv_layer(
            self.conv_cfg,
            self.high_in_channels,
            self.out_channels,
            kernel_size=1,
            bias=False)
        self.flow_conv = build_conv_layer(
            self.conv_cfg,
            out_channels * 2,
            2,
            kernel_size=3,
            padding=1,
            bias=False)

    def forward(self, low_x, high_x):
        """Forward function."""
        low_x = self.low_conv(low_x)
        low_feat = self.low_down_conv(low_x)
        high_feat = self.high_down_conv(high_x)
        high_feat = F.interpolate(
            high_feat,
            low_x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([low_feat, high_feat], dim=1)
        flow = self.flow_conv(output)
        high_feat = FAM.warp(high_x, flow)
        output = low_x + high_feat
        return output

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w*2, h*2]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h*2).view(-1, 1).repeat(1, w*2)
        row = torch.linspace(-1.0, 1.0, w*2).repeat(h*2, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid)
        return output


@NECKS.register_module()
class SFNeck(nn.Module):
    """Semantic Flow Network.

    This neck is the implementation of
    `SFNet <https://arxiv.org/abs/2002.10120>`_.

    Args:
        in_channels (Sequence(int)): Input channels.
        channels (int): Channels after modules, before conv_seg.
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        out_indices (tuple): Tuple of indices of output.
            Often set to (0,1,2,3,4) to enable aux. heads.
            Default: (0, 1, 2, 3, 4).
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_channels,
                 channels,
                 pool_scales=(1, 2, 3, 6),
                 align_corners=False,
                 out_indices=(0, 1, 2, 3, 4),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(SFNeck, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(pool_scales, (list, tuple))
        self.in_channels = in_channels
        self.channels = channels
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.fp16_enabled = False

        self.ppm = PPM(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.fams = []
        self.convs = []
        for in_channels in self.in_channels[:-1]:
            self.fams.append(FAM(
                in_channels,
                self.channels,
                self.channels // 2,
                align_corners=self.align_corners,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
            self.convs.append(ConvModule(
                self.channels,
                self.channels,
                1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.fams = nn.ModuleList(self.fams)
        self.convs = nn.ModuleList(self.convs)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        psp_outs = [inputs[-1]]
        psp_outs.extend(self.ppm(inputs[-1]))
        psp_out = torch.cat(psp_outs, dim=1)
        psp_out = self.bottleneck(psp_out)

        out = psp_out
        outs = [psp_out]
        fusion_outs = [psp_out]
        for i in reversed(range(len(inputs) - 1)):
            out = self.fams[i](inputs[i], out)
            outs.append(out)
            fusion_outs.append(self.convs[i](out))
        fusion_outs.reverse()
        for i in range(1, len(fusion_outs)):
            fusion_outs[i] = F.interpolate(
                fusion_outs[i],
                fusion_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fusion_out = torch.cat(fusion_outs, dim=1)
        outs.append(fusion_out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
