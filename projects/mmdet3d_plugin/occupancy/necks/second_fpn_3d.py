# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS
import torch.nn.functional as F
import time
import pdb
from .attention_3d import *

class ASPP_3D(nn.Module):
    def __init__(self, in_channel=32, depth=16):
        super(ASPP_3D, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv3d(in_channel, depth, 3, 1, padding=1, dilation=1)
        self.atrous_block3 = nn.Conv3d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block6 = nn.Conv3d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv3d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv3d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv3d(depth * 5, in_channel, 1, 1)
    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([ atrous_block1,  atrous_block3, atrous_block6,
                                            atrous_block12, atrous_block18   ], dim=1))
        return net


@NECKS.register_module()
class SECONDFPN3D(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """
    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 upsample_cfg=dict(type='deconv3d', bias=False),
                 conv_cfg=dict(type='Conv3d', bias=False),
                 use_conv_for_no_stride=False,
                 use_output_upsample=False,
                 with_cp=False,
                 init_cfg=None):

        # replacing GN with BN3D, performance drops from 42.5 to 40.9.
        # the difference may be exaggerated because the performance can fluncate a lot

        super(SECONDFPN3D, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.with_cp = with_cp

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(
                upsample_layer, build_norm_layer(norm_cfg, out_channel)[1], nn.ReLU(inplace=True))

            deblocks.append(deblock)

        self.deblocks = nn.ModuleList(deblocks)

        self.use_output_upsample = use_output_upsample
        if self.use_output_upsample:
            output_channel = sum(out_channels)
            self.output_deblock = nn.Sequential(
                build_upsample_layer(
                    upsample_cfg, in_channels=output_channel,
                    out_channels=output_channel, kernel_size=2, stride=2),
                build_norm_layer(norm_cfg, output_channel)[1],
                nn.ReLU(inplace=True),

            )

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

        # * WVA 的可学习门控系数 α：初始化为 0，避免训练初期不可靠的时序信息干扰 Vvox
        self.alpha = nn.Parameter( torch.zeros(1) )
        # * 3D 交叉注意力：查询来自 Vvox，键和值来自可靠时序体素 Ṽtem
        self.attention_3d = LinearAttention3D( query_dim=384, dim=384,  heads=2 )



    @auto_fp16()
    def forward(self, x, depth, temporal_voxel=None):
        """Fuse multi-scale voxel features and temporal context.

        Args:
            x (list[torch.Tensor]): Backbone features in (B, C, X, Y, Z).
            depth (torch.Tensor): Depth probabilities; unused here.
            temporal_voxel (list[torch.Tensor]): Temporal voxel features.
                The first element is required by the attention block.

        Returns:
            list[torch.Tensor]: A single fused voxel feature map.
        """
        # *==============================================#
        # * 5.2.1 对齐各尺度的空间分辨率与通道数
        # x 是主干输出的三级特征列表；以下 shape 对应 temporal_baseline_custom.py，后三维是 XYZ
        # x[0]=[B,128,128,128,16]，x[1]=[B,256,64,64,8]，x[2]=[B,512,32,32,4]
        # deblocks 在 __init__ 中构建：转置 3D 卷积 + 归一化 + ReLU，当前上采样倍率分别为 1、2、4
        # 每路统一输出 [B,128,128,128,16]；倍率为 1 的分支保持网格尺寸，仍进行可学习的特征变换
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        # *----------------------------------------------#
        # * 5.2.2 沿通道拼接多尺度主分支特征
        # 三路各 128 通道 -> out=[B,384,128,128,16]；保留不同尺度的信息，不在此对各路求和
        # 若只有一路则直接使用该路；当前配置为三路
        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]

        # *----------------------------------------------#
        # * 5.2.3 可选的额外输出上采样
        # 当前 use_output_upsample=False，跳过；启用时 XYZ 尺寸各扩大 2 倍、通道数不变
        # checkpoint 通过反向传播时重算该模块，减少需要保存的中间激活
        if self.use_output_upsample:
            out = torch.utils.checkpoint.checkpoint(self.output_deblock, out)

        # *----------------------------------------------#
        # * 5.2.4 WVA：通过体素交叉注意力检索时序内容，并残差融合
        # 实现：同目录 attention_3d.py 的 LinearAttention3D.forward()，由本文件导入；不是 image2bev 中的同名类
        # query 来自主分支 out，key/value 来自 temporal_voxel[0]；当前两者均为 [B,384,128,128,16]
        # 主分支决定需要检索什么，时序分支提供内容；注意力输出与 out 同形状，可直接残差相加
        # alpha 为可学习标量，初始化为 0，使该融合初始输出保持主分支值，再由训练学习时序贡献
        # alpha 并非固定比例或概率，代码没有限制其取值范围；depth 参数在本方法中未参与计算
        out = self.alpha * self.attention_3d( query = out,  x = temporal_voxel[0] ) + out  # Vret = alpha * CrossAtt(Vvox, Vtem) + Vvox


        # *----------------------------------------------#
        # * 5.2.5 返回单元素特征列表，供占用预测头使用
        # [out] 中 out=[B,384,128,128,16]；列表维度不是 batch 或时间维，也不再包含三个尺度
        return [out]
