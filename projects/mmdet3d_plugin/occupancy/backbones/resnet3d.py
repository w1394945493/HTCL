import math
from functools import partial
from mmdet3d.models.builder import BACKBONES
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule
from .crp3d import CPMegaVoxels

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_cfg=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_cfg=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = build_norm_layer(norm_cfg, planes * self.expansion)[1]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

@BACKBONES.register_module()
class CustomResNet3D(BaseModule):
    def __init__(self,
                 depth,
                 block_inplanes=[64, 128, 256, 512],
                 block_strides=[1, 2, 2, 2],
                 out_indices=(0, 1, 2, 3),
                 num_stage=4,
                 n_input_channels=3,
                 shortcut_type='B',
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 crp3d=False,
                 crp_level=2,
                 widen_factor=1.0):
        super().__init__()
        
        layer_metas = {
            10: [1, 1, 1, 1],
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
        }
        
        if depth in [10, 18, 34]:
            block = BasicBlock
        else:
            assert depth in [50, 101]
            block = Bottleneck
        
        layers = layer_metas[depth]
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.out_indices = out_indices
        self.num_stage = num_stage
        
        # replace the first several downsampling layers with the channel-squeeze layers
        self.input_proj = nn.Sequential(
            nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), bias=False),
            build_norm_layer(norm_cfg, self.in_planes)[1],
            nn.ReLU(inplace=True),
        )
        
        self.layers = nn.ModuleList()
        for i in range(len(block_inplanes)):
            if (i + 1) > self.num_stage:
                break
            
            self.layers.append(self._make_layer(block, block_inplanes[i], layers[i], 
                                shortcut_type, block_strides[i], norm_cfg=norm_cfg))
            
        # optional: build crp3d
        self.crp3d = crp3d
        if self.crp3d:
            self.crp_level = crp_level
            crp_in_channel = block_inplanes[self.crp_level]
            self.CP_mega_voxels = CPMegaVoxels(
                crp_in_channel, (32, 32, 4), bn_momentum=0.1,
            )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.forward_dic = {}

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, norm_cfg=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1])

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  norm_cfg=norm_cfg))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, norm_cfg=norm_cfg))

        return nn.Sequential(*layers)

    def crp_loss(self, CP_mega_matrices):
        P_logits = self.forward_dic['crp_logits']
        loss_rel_ce = compute_super_CP_multilabel_loss(P_logits, CP_mega_matrices)
        
        return loss_rel_ce

    def forward(self, x):  
        # *==============================================#
        # * 5.1.1 输入投影：将主分支体素映射到三维主干的起始通道数
        # 输入按 [B,C,X,Y,Z] 排列；以下 shape 对应 temporal_baseline_custom.py，不是所有配置的固定尺寸
        # input_proj 为 1x1x1 Conv3d + 归一化 + ReLU，改变特征表示但不下采样空间网格
        # 当前输入/输出均为 [B,128,128,128,16]；通道数相同仍会经过可学习的通道混合
        x = self.input_proj(x)
        res = []
        # *----------------------------------------------#
        # * 5.1.2 逐级执行三维残差编码，提取多尺度空间上下文
        # 当前 depth=18、num_stage=3，每级包含两个 BasicBlock；三级步长分别为 1、2、2
        # stage 0：[B,128,128,128,16] -> [B,128,128,128,16]
        # stage 1：[B,128,128,128,16] -> [B,256,64,64,8]
        # stage 2：[B,256,64,64,8] -> [B,512,32,32,4]
        # 后级增加通道并降低 XYZ 分辨率，以聚合更大范围的上下文；此处尚未融合 temporal_voxel
        for index, layer in enumerate(self.layers):
            x = layer(x)

            # *----------------------------------------------#
            # * 5.1.3 可选 CRP 分支：在指定层级建模三维上下文关系
            # 当前配置使用默认 crp3d=False，不执行此分支；启用时更新特征并保存关系预测供外部损失使用
            if self.crp3d and (self.crp_level == index):
                outputs = self.CP_mega_voxels(x)
                self.forward_dic['crp_logits'] = outputs['P_logits']
                x = outputs['x']

            # *----------------------------------------------#
            # * 5.1.4 收集需要输出的层级特征
            # 当前 out_indices=(0,1,2)，保留全部三级输出；res 是列表，不在这里拼接或统一分辨率
            if index in self.out_indices:
                res.append(x)

        # *----------------------------------------------#
        # * 5.1.5 返回多尺度特征列表，交给 SECONDFPN3D 进行上采样、拼接和时序融合
        # res[0]=[B,128,128,128,16]，res[1]=[B,256,64,64,8]，res[2]=[B,512,32,32,4]
        return res

# def generate_model(model_depth, **kwargs):
#     assert model_depth in [10, 18, 34, 50, 101, 152, 200]

#     if model_depth == 10:
#         model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
#     elif model_depth == 18:
#         model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
#     elif model_depth == 34:
#         model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
#     elif model_depth == 50:
#         model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
#     elif model_depth == 101:
#         model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
#     elif model_depth == 152:
#         model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
#     elif model_depth == 200:
#         model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

#     return model


def compute_super_CP_multilabel_loss(pred_logits, CP_mega_matrices):
    logits = []
    labels = []
    bs, n_relations, _, _ = pred_logits.shape
    for i in range(bs):
        pred_logit = pred_logits[i, :, :, :].permute(
            0, 2, 1
        )  # n_relations, N, n_mega_voxels
        CP_mega_matrix = CP_mega_matrices[i]  # n_relations, N, n_mega_voxels
        logits.append(pred_logit.reshape(n_relations, -1))
        labels.append(CP_mega_matrix.reshape(n_relations, -1))

    logits = torch.cat(logits, dim=1).T  # M, 4
    labels = torch.cat(labels, dim=1).T  # M, 4

    cnt_neg = (labels == 0).sum(0)
    cnt_pos = labels.sum(0)
    pos_weight = cnt_neg / cnt_pos
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_bce = criterion(logits, labels.float())
    
    return loss_bce
