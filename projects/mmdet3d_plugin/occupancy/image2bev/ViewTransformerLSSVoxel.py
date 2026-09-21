# Copyright (c) Phigent Robotics. All rights reserved.
import math
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
from mmdet3d.ops.bev_pool import bev_pool
from mmdet3d.ops.voxel_pooling import voxel_pooling
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast
from projects.mmdet3d_plugin.utils.gaussian import generate_guassian_depth_target
from mmdet.models.backbones.resnet import BasicBlock
from projects.mmdet3d_plugin.utils.semkitti import semantic_kitti_class_frequencies, kitti_class_names, CE_ssc_loss
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from PIL import Image
import pdb
from .ViewTransformerLSSBEVDepth import *
from .semkitti_depthnet import SemKITTIDepthNet
from .temporal_retrieve  import *
norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)
from .gwc_encoder import *
from .LEAStereo.LEAStereo import LEA_encoder
from .manydepth.temporal_encoder import temporal_encoder


class volume_interaction(nn.Module):
    def __init__(self,  out_channels=1):
        super(volume_interaction, self).__init__()
        self.dres1 = nn.Sequential(convbn_3d(2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   hourglass(32))
        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.out3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(32, 1, 3, 1, 1))
    def forward(self, stereo_volume, lss_volume):
        # *========================================================#
        # * 拼接双目代价体与 LSS 深度分布，并通过 3D Hourglass 网络融合
        stereo_volume=stereo_volume.unsqueeze(1)
        lss_volume=lss_volume.unsqueeze(1)
        all_volume=torch.cat( (stereo_volume, lss_volume ), dim=1)
        data1_ = self.dres1(all_volume)
        data2_ = self.dres2(data1_)
        data3 = self.dres3(data2_) + data1_
        data3 =  self.out3(data3)
        data3 = data3.squeeze(1)
        data3 = F.softmax(data3, dim=1)
        data1, data2=None, None
        return data3, [data1, data2]


class temporal_interaction(nn.Module):
    def __init__(self,  out_channels=1):
        super(volume_interaction, self).__init__()
        self.dres1 = nn.Sequential(convbn_3d(2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   hourglass(32))
        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.out3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(32, 1, 3, 1, 1))
    def forward(self, stereo_volume, lss_volume):
        stereo_volume=stereo_volume.unsqueeze(1)
        lss_volume=lss_volume.unsqueeze(1)
        all_volume=torch.cat( (stereo_volume, lss_volume ), dim=1)
        data1_ = self.dres1(all_volume)
        data2_ = self.dres2(data1_)
        data3 = self.dres3(data2_) + data1_
        data3 =  self.out3(data3)
        data3 = data3.squeeze(1)
        data3 = F.softmax(data3, dim=1)
        data1, data2=None, None
        return data3, [data1, data2]


@NECKS.register_module()
class ViewTransformerLiftSplatShootVoxel(ViewTransformerLSSBEVDepth):
    def __init__(
            self,
            loss_depth_weight,
            semkitti=False,
            imgseg=False,
            imgseg_class=20,
            lift_with_imgseg=False,
            point_cloud_range=None,
            loss_seg_weight=1.0,
            loss_depth_type='bce', ##'bce', smooth
            point_xyz_channel=0,
            point_xyz_mode='cat',
            depth_model="lea",
            temporal_num = 4,
            pose_pretrained=None,
            **kwargs,
        ):

        super(ViewTransformerLiftSplatShootVoxel, self).__init__(loss_depth_weight=loss_depth_weight, **kwargs)

        self.leamodel=LEA_encoder(maxdisp=192)
        self.volume_interaction = volume_interaction()
        self.temporal_deformable = multipatch_deformable( indim=3, outdim=3 )
        # * CPA：分别提取当前帧 2D pattern 与对齐历史帧 3D pattern，用于计算跨帧亲和度
        self.curr_patch = multi_patch2d(in_channel=64, depth=1)
        self.warped_patch = multi_patch3d(in_channel=3, depth=1)
        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.temporal_encoder = temporal_encoder(
            maxdisp=112,
            height=384,
            width=1280,
            pose_pretrained=pose_pretrained,
        )
        self.temporal_prehourglass = nn.Sequential(convbn_3d( temporal_num-1, 32, 3, 1, 1),  nn.ReLU(inplace=True),
                                                hourglass(32),
                                                convbn_3d(32, 64, 3, 1, 1),
                                                nn.ReLU(inplace=True),
                                                nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=1, bias=False))

        self.temporal_hourglass = nn.Sequential(convbn_3d( 64, 32, 3, 1, 1),  nn.ReLU(inplace=True),
                                                hourglass(32),
                                                convbn_3d(32, 64, 3, 1, 1),
                                                nn.ReLU(inplace=True),
                                                nn.Conv3d(64, 384, kernel_size=3, padding=1, stride=1, bias=False))

        self.semkitti = semkitti

        self.loss_depth_type = loss_depth_type
        self.cam_depth_range = self.grid_config['dbound']
        self.constant_std = 0.5
        self.point_cloud_range = point_cloud_range

        ''' Extra input for Splating: except for the image features, the lifted points should also contain their positional information '''
        self.point_xyz_mode = point_xyz_mode
        self.point_xyz_channel = point_xyz_channel

        assert self.point_xyz_mode in ['cat', 'add']
        if self.point_xyz_mode == 'add':
            self.point_xyz_channel = self.numC_Trans

        if self.point_xyz_channel > 0:
            assert self.point_cloud_range is not None
            self.point_cloud_range = torch.tensor(self.point_cloud_range)

            mid_channel = self.point_xyz_channel // 2
            self.point_xyz_encoder = nn.Sequential(
                nn.Linear(in_features=3, out_features=mid_channel),
                nn.BatchNorm1d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=mid_channel, out_features=self.point_xyz_channel),
            )

        ''' Auxiliary task: image-view segmentation '''
        self.imgseg = imgseg
        if self.imgseg:
            self.imgseg_class = imgseg_class
            self.loss_seg_weight = loss_seg_weight
            self.lift_with_imgseg = lift_with_imgseg

            # build a small segmentation head
            in_channels = self.numC_input
            self.img_seg_head = nn.Sequential(
                BasicBlock(in_channels, in_channels),
                BasicBlock(in_channels, in_channels),
                nn.Conv2d(in_channels, self.imgseg_class, kernel_size=1, padding=0),
            )

        self.forward_dic = {}

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape  ## [1, 1, 384, 1280]
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample) #
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  #
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]

        return gt_depths_vals, gt_depths.float()

    def get_diff_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1)[:, :, :, 1:]

        mask = torch.max(gt_depths, dim=3).values > 0.0
        mask = mask.unsqueeze(3)
        gt_depths = gt_depths * mask

        gt_depths = gt_depths.permute(0, 3, 1, 2).contiguous()
        gt_depths = gt_depths.unsqueeze(1)

        return gt_depths.float()

    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        return depth_loss

    @force_fp32()
    def get_smooth_depth_loss(self, depth_labels, depth_preds):

        B,D,H,W = depth_preds.shape
        depth_labels =  F.interpolate(depth_labels, [ H, W], mode='bilinear', align_corners=False)

        with torch.cuda.device_of(depth_preds):
            disp = torch.reshape(torch.arange(0, D, device=torch.cuda.current_device(), dtype=torch.float32),[1,D,1,1])
            disp = disp.repeat(depth_preds.size()[0], 1, depth_preds.size()[2], depth_preds.size()[3])
            depth_preds = torch.sum(depth_preds * disp, 1).unsqueeze(1)

        mask = (depth_labels > 0)
        mask.detach_()
        loss = F.smooth_l1_loss(depth_preds[mask], depth_labels[mask], reduction='mean')
        return loss

    @force_fp32()
    def get_klv_depth_loss(self, depth_labels, depth_preds):
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
            self.downsample, self.cam_depth_range, constant_std=self.constant_std)

        depth_values = depth_values.view(-1)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2]))

        depth_gaussian_labels = depth_gaussian_labels.view(-1, self.D)[fg_mask]
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)[fg_mask]

        depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean', log_target=False)

        return depth_loss

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if self.loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)

        elif self.loss_depth_type == 'kld':
            depth_loss = self.get_klv_depth_loss(depth_labels, depth_preds)

        elif self.loss_depth_type == 'smooth':
            depth_loss = self.get_smooth_depth_loss(depth_labels, depth_preds)

        else:
            pdb.set_trace()

        return self.loss_depth_weight * depth_loss

    @force_fp32()
    def get_seg_loss(self, seg_labels):
        class_weights = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001)).type_as(seg_labels).float()
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=0, reduction="mean",
        )
        seg_preds = self.forward_dic['imgseg_logits']
        if seg_preds.shape[-2:] != seg_labels.shape[-2:]:
            seg_preds = F.interpolate(seg_preds, size=seg_labels.shape[1:])

        loss_seg = criterion(seg_preds, seg_labels.long())

        return self.loss_seg_weight * loss_seg

    def voxel_pooling(self, geom_feats, x):
        # *==============================================#
        # * 体素汇聚总述：按三维坐标将视锥采样点的特征累加到 XYZ 体素网格
        # 主分支 4.6 与时序分支 4.10 共用此方法；输入坐标已经由 get_geometry() 计算，此处不再做相机投影
        # x=[B,N,D,H,W,C] 保存每个视锥采样点的特征，geom_feats=[B,N,D,H,W,3] 保存其三维坐标
        # N 是相机视角数，不是历史帧数；D 是深度假设数，H、W 是图像特征图尺寸
        # 输出为 [B,C_out,X,Y,Z]；多个采样点落入同一体素时按通道求和，不是取平均或选择唯一深度
        # *----------------------------------------------#
        # * （1）展平视锥特征，让每一行对应一个三维采样点
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # [B,N,D,H,W,C] -> [Nprime,C]；坐标随后按相同顺序展平，保持点与特征一一对应
        x = x.reshape(Nprime, C)

        # *----------------------------------------------#
        # * （2）将连续三维坐标转换为离散体素索引，并附加 batch 编号
        # dx 是 XYZ 方向的体素尺寸，bx 是首个体素中心，bx-dx/2 是网格下边界；nx 是各轴体素数量
        # (坐标-下边界)/体素尺寸得到网格位置；long() 向零截断，非负范围内等价于向下取整
        # 注意：略低于下边界的负小数也会被截成 0；下方索引过滤并非严格的连续坐标边界判断
        geom_xyz = geom_feats.clone()
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        # batch_ix=[Nprime,1]；追加后每行是 [ix,iy,iz,batch_id]，防止不同样本的同坐标特征混在一起
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # *----------------------------------------------#
        # * （3）根据 XYZ 索引过滤网格外的采样点
        # kept=[Nprime]；保留 M 个点后，x=[M,C]、geom_feats=[M,4]，两者使用同一个掩码
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # *----------------------------------------------#
        # * （4）可选：将连续三维位置编码为特征，与输入内容融合
        # 使用量化前的 geom_xyz，并应用相同掩码；未启用此分支时 C_out=C
        if self.point_xyz_channel > 0:
            geom_xyz = geom_xyz.view(Nprime, 3)
            geom_xyz = geom_xyz[kept]

            # 按 point_cloud_range 将范围内的 XYZ 坐标归一化到 [-1,1]，再用网络编码为位置特征
            pc_range = self.point_cloud_range.type_as(geom_xyz) # normalize points to [-1, 1]
            geom_xyz = (geom_xyz - pc_range[:3]) / (pc_range[3:] - pc_range[:3])
            geom_xyz = (geom_xyz - 0.5) * 2
            geom_xyz_feats = self.point_xyz_encoder(geom_xyz)

            if self.point_xyz_mode == 'cat':
                # 沿通道拼接内容与位置特征：[M,C] + [M,C_xyz] -> [M,C+C_xyz]
                x = torch.cat((x, geom_xyz_feats), dim=1)

            elif self.point_xyz_mode == 'add':
                # 逐元素相加，位置特征通道需要与内容兼容；输出通道数保持不变
                x += geom_xyz_feats

            else:
                raise NotImplementedError

        # *----------------------------------------------#
        # * （5）调用 BEV pooling 算子累加同一体素内的特征，并调整输出轴顺序
        # 算子来自 mmdet3d.ops.bev_pool；按 batch_id 和 XYZ 索引汇聚，无采样点的体素保持零
        # 输入尺寸参数顺序为 Z、X、Y，输出 [B,C_out,Z,X,Y]；这里保留高度维，并未压成二维 BEV
        final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = final.permute(0, 1, 3, 4, 2)  # [B,C_out,Z,X,Y] -> [B,C_out,X,Y,Z]

        # 当前网格为 128x128x16；未拼接位置特征时，主分支输出 128 通道，时序分支输出 64 通道
        return final

    def forward(self, input, gt, mode, imgl, imgr, left_input, right_input  ):

        # *=============================================#
        # * 4.1 解包图像特征与相机参数，准备当前帧双目图像
        # 以下 shape 对应当前配置：N=1、T=4、D=112、上下文通道数=128；行内示例取 B=1
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]

        B, N, C, H, W = x.shape # [B, 1, 640, 48, 160]，H/W 是特征图分辨率
        x = x.view(B * N, C, H, W)
        calib = input[16]  # 双目标定量：焦距与基线的乘积

        # 从时序队列取最后一帧，移除单视角维，再缩放到双目网络的输入分辨率
        if  imgl.shape[1]>1:
            imgl, imgr = imgl[:, -1, ...], imgr[:, -1, ...] # (1 1 3 384 1280)，当前帧左右图像
        imgl, imgr = F.interpolate(imgl.squeeze(1), size=[288, 960], mode='bilinear', align_corners=True), F.interpolate(imgr.squeeze(1), size=[288, 960], mode='bilinear', align_corners=True) # (1 3 288 960)

        # *---------------------------------------------#
        # * 4.2 双目深度估计：使用 LEAStereo 构建当前帧深度代价体
        # 将代价体插值到 [B, 112, H, W]，对负代价做 softmax 得到深度概率
        stereo_volume = self.leamodel(imgl, imgr, calib )["classfy_volume"] # (1 1 64 96 320)
        stereo_volume = F.interpolate(stereo_volume, size=[ 112, H, W ], mode='trilinear', align_corners=True).squeeze(1) # (1 112 48 160)
        stereo_volume = F.softmax(-stereo_volume, dim=1) # (1 112 48 160)

        # 可选辅助分支：预测图像语义，供分割监督和后续特征拼接使用
        if self.imgseg:
            self.forward_dic['imgseg_logits'] = self.img_seg_head(x)
        # *---------------------------------------------#
        # * 4.3 单目深度估计：预测离散深度分布与图像上下文特征
        # depth_net 接收当前左图特征和相机条件向量，输出 D 个深度通道及 numC_Trans 个上下文通道
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...] # (1 112 48 160)
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...] # (1 128 48 160)，每个像素的上下文特征
        depth_prob = self.get_depth_dist(depth_digit) # (1 112 48 160)，每个像素的离散深度分布

        # *---------------------------------------------#
        # * 4.4 融合双目与单目深度信息
        # 更新 depth_prob，供后续 Lift 和外部深度监督使用；auxility 在此函数中未继续使用
        depth_prob, auxility = self.volume_interaction(stereo_volume, depth_prob)

        # 可选：将图像语义概率拼接到上下文特征；下方 128 通道示例对应未拼接时的配置
        if self.imgseg and self.lift_with_imgseg:
            img_segprob = torch.softmax(self.forward_dic['imgseg_logits'], dim=1)
            img_feat = torch.cat((img_feat, img_segprob), dim=1)
        # *---------------------------------------------#
        # * 4.5 Lift：用深度概率加权图像上下文，构建视锥特征体
        # 将每个像素的上下文沿 D 个深度位置展开，再整理为 [B, N, D, H, W, C] 供体素汇聚使用
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2) # (1 128 112 48 160)
        volume = volume.view(B, N, -1, self.D, H, W) # (1 1 128 112 48 160)
        volume = volume.permute(0, 1, 3, 4, 5, 2) # (1 1 112 48 160 128)
        # *---------------------------------------------#
        # * 4.6 Splat：根据相机几何关系将视锥特征汇聚到主分支体素
        # geom 给出视锥采样点的三维坐标；bev_feat 为 [B, 128, X, Y, Z]，网格为 128x128x16
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda) # (1 1 112 48 160 3)
        bev_feat = self.voxel_pooling(geom, volume) # (1 128 128 128 16)

        # *=============================================#
        # * 4.7-4.10 时序分支总述：几何组织候选信息，学习细化历史内容，再按相关性加权并融合
        # 以下按“问题 -> 所需能力 -> 模块”理解当前代码，并非断言作者实际的构思顺序。
        # 4.7：相机运动使跨帧同一像素不再对应同一物体，而投影所需的深度未知，因此先建立多个候选对应。
        # 用相对位姿、内参和多个深度假设对齐历史特征，再按历史帧堆叠；不急于选出唯一深度或生成深度概率。
        # 得到 [B,N,D,H,W]：D、H、W 构成视锥特征网格，N 张历史帧的响应作为通道；此时只是组织了候选信息。
        # 4.8：几何对齐仍可能存在位姿误差、物体运动等造成的残余错位，需要更灵活地利用初始位置附近的内容。
        # ADR 内容分支在相邻深度层间聚合信息，并学习高度、宽度方向的采样偏移；三级串联后融合不同层级输出。
        # 4.9：采到历史内容不等于它与当前帧相关，因此引入当前特征作为比较依据，单独计算内容的相关性权重。
        # CPA 比较当前帧与原始对齐历史体的多尺度局部模式，将得到的亲和度乘到 ADR 输出上。
        # 两条路径分工：历史分支提供内容，当前帧参与评价相关性；当前实现中 CPA 权重不直接输入 ADR 偏移预测器。
        # 4.10：加权后的内容仍在深度假设与图像坐标组成的网格上，需要转换为占用预测使用的 XYZ 空间表示。
        # 因此先编码时序内容，再按几何关系汇聚并编码为 temporal_voxel；4.11 将其与主分支结果一起返回。
        # 外部 bev_encoder 通过 WVA 交叉注意力将时序体素融入主分支；训练时还对时序体素单独施加占用监督。
        # “简单堆叠”的局限不是拼接操作本身，而是组织信息后缺少进一步的对应关系度量和内容细化。
        # 值得学习的设计方法：先拆清“对应、采样、相关性、空间表示”四个问题，再让不同模块承担明确职责。
        # 用几何先验建立初始对应，以可学习模块处理剩余误差；在不确定时保留候选信息，供后续任务逐步利用。
        # 将“提供内容”与“评价相关性”分开，并检查各阶段的坐标、通道含义和接口，而不只检查 shape 是否匹配。
        # 这些问题并不唯一决定当前方案；卷积层数、通道数、均值压缩及权重形式仍需通过消融实验验证。
        # 理解边界：通道均值会丢失信息；亲和度可为负，不是可信概率；相同 D 长度不保证两分支的深度采样一致。

        # *=============================================#
        # * 4.7 构建对齐时序体：将历史左图信息对齐到当前参考帧
        # 实现入口：同目录 manydepth/temporal_encoder.py 的 temporal_encoder.forward()
        # ref = reference（参考帧）：以当前帧的像素网格为对齐基准，最终预测也对应当前帧
        # sour = source（来源帧）：历史帧提供待采样的图像特征，通过几何变换对齐到参考帧网格
        # 命名描述对齐角色，而非固定时间属性；这里选择当前帧作 reference、历史帧作 source
        # 具体采用反向采样：从 ref 的像素和候选深度出发，算出 sour 中的采样位置，再读取历史特征
        # left_input 是 [B, T, 384, 1280, 3] 的 RGB 像素序列；拆出当前帧ref与 T-1 个历史帧sour
        # 当前 T=4，按 [t-3, t-2, t-1, t] 排列；permute 将 RGB 通道移到空间维前，cuda 将数据放到 GPU
        img_left_ref, img_left_sour = left_input[:, -1, ...].unsqueeze(1).permute(0,1,4,2,3).cuda(), left_input[:,:-1, ...].permute(0,1,4,2,3).cuda()  # (1 1 3 384 1280) (1 3 3 384 1280)
        # 内部逐张处理历史图：冻结的 PoseNet 在 no_grad 下估计其与当前帧之间的相对旋转和平移
        # 独立的 ResNet18 提取当前/历史图像的 1/4 尺度特征 [B, 64, 96, 320]
        # 对每个候选深度：当前像素反投影为三维点 -> 相对位姿变换 -> 投影到历史图 -> grid_sample 采样
        # 几何对齐实现：manydepth/networks/resnet_encoder.py 的 ResnetEncoderMatching.match_features()
        # 对齐结果沿 64 个特征通道求均值，再堆叠各历史帧，得到 [B, T-1, D, 96, 320]
        # curr_feature=[B,64,96,320]；batch_waped_feature=[B,3,112,96,320]，3 是历史帧数，112 是深度假设数
        # curr_feature 保留当前帧的 64 维特征；batch_waped_feature 是“按深度假设组织的历史对齐特征体”
        # 112 维索引候选深度，实际深度值存于匹配编码器的 depth_bins[d]；张量元素存的是特征响应，不是深度值
        # batch_waped_feature[b,i,d,h,w]：假设当前像素 (h,w) 深度为 depth_bins[d]，
        # 在第 i 张历史图的对应位置采样 64 维特征，再沿通道取均值，得到一个标量响应
        # 单张历史图：[B,64,h,w] -> 按 D 个深度对齐 [B,D,64,h,w] -> 通道均值 [B,D,h,w]
        # 再堆叠 T-1 张历史图得到 [B,T-1,D,h,w]；此处 h=96、w=320
        # 每个深度切片保留该假设下的历史信息；尚未计算它与当前特征的相似度，也未判断哪个深度最可信
        # 此输出是按深度假设组织的历史特征，不是归一化深度概率，也尚未汇聚到 XYZ 体素网格
        # 注意：时序编码器当前在 0~112 范围生成 112 个候选深度，与主分支 [2,58)、步长 0.5 的采样不同
        curr_feature, batch_waped_feature = self.temporal_encoder( ref_images=img_left_ref, source_images=img_left_sour, intrinsics=intrins ) # (1 64 96 320) (1 3 112 96 320)


        # 两路分工：curr_feature 提供当前内容作为比较依据；batch_waped_feature 提供历史内容及比较信息
        # 历史体分成 ADR 内容细化与 CPA 亲和度计算两条路径，最后用 CPA 权重调制 ADR 输出
        # 将空间分辨率统一到 H=48、W=160；当前 D=112 不变，插值不会将时序深度采样自动校准为主分支的米制位置
        curr_feature = F.interpolate(curr_feature, size=[H, W], mode='bilinear', align_corners=True) # (1 64 96 320)->(1 64 48 160)
        batch_waped_feature = F.interpolate(batch_waped_feature, size=[self.D, H, W], mode='trilinear', align_corners=True) # (1 3 112 48 160)
        # *---------------------------------------------#
        # * 4.8 ADR（Affinity-based Dynamic Refinement，基于亲和度的动态细化）：先计算可变形内容分支
        #---------------------------------------------------------------------------------
        # 论文依据：HTCL（ECCV 2024）第 3.4/3.5 节，CPA 度量跨帧对应关系，ADR 结合亲和度与可学习采样偏移
        # https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00502.pdf
        # ADR 的目标是利用相关位置及其邻域内容补充不完整观测；不只是把历史帧直接堆叠起来
        # 实现：同目录 temporal_retrieve.py 的 multipatch_deformable.forward()
        # 本地实现将 ADR 拆开计算：这里仅产生细化内容，亲和度加权在下方 CPA 计算后完成
        # 此调用没有接收 CPA 权重，不能理解为将权重显式输入偏移预测器；应以实际调用关系为准
        # 仅处理历史体，不使用 curr_feature；三级可变形卷积在 H/W 方向调整采样，再拼接融合

        #---------------------------------------------------------------------------------
        # 细化动机：几何对齐依赖估计位姿和候选深度，位姿误差、物体运动及遮挡都可能使历史采样响应不可靠
        # 可学习偏移允许在初始对齐位置附近寻找有用内容，例如从车辆边缘旁的背景移向边缘特征；不保证纠正所有误差
        # dimension='HW' 限定可学习偏移沿图像空间方向；3x3x3 卷积仍会聚合相邻 D 层、空间邻域及历史通道
        # 因而它既调整空间采样，也组合邻近深度假设下的响应，但不直接将 112 个候选转换为深度概率或做 argmax

        #---------------------------------------------------------------------------------
        # [B,3,D,H,W] -> 各级 8 通道特征 -> 拼接为 24 通道 -> 融合为 [B,3,D,H,W]
        # 输入 3 通道对应三张历史帧；卷积已混合帧间信息，输出 3 通道不再逐一对应某张历史帧
        # 为什么通道均值后的响应仍能细化：batch_waped_feature 存的是图像特征的聚合响应，不是候选编号列表
        # 固定历史帧 i 和深度 d，batch_waped_feature[b,i,d,:,:] 是一张 HxW 响应图，仍可能保留边缘和区域变化
        # 单个位置虽只有一个标量，周围空间位置、相邻深度层及不同历史帧的响应共同组成可学习的局部模式
        # 卷积利用这些模式生成新特征；可变形采样进一步允许在规则网格附近调整取值位置，利用偏移的有用响应
        # 例如初始对齐使边缘响应偏移，网络可学习从附近读取它；这里采样的是对齐特征体，不是重新读取历史 RGB 图
        # 这不是重新估计位姿或选择深度；也不能恢复此前 64 通道求均值丢失的细节，效果依赖剩余信息和训练
        # 分工：此分支生成细化后的历史内容；下方 CPA 利用当前帧与历史帧的模式计算权重，再对这些内容加权
        defomable_batch_waped_feature = self.temporal_deformable(batch_waped_feature) # [B,3,D,H,W]

        # *---------------------------------------------#
        # * 4.9 CPA（Cross-frame Pattern Affinity，跨帧模式亲和度）：度量对应关系并加权 ADR 内容
        #---------------------------------------------------------------------------------
        # 目的：衡量历史局部内容与当前帧是否相关，为可靠时序信息的聚合提供权重
        # 论文使用多组多尺度上下文和去均值的余弦度量；下方说明的是当前仓库的具体实现
        # 实现：同目录 temporal_retrieve.py 的 multi_patch2d / multi_patch3d
        # CPA 使用 ADR 之前的历史体，与 ADR 并行处理；比较的是学习后的局部模式，不是原始 64 维特征
        # 两路均用 dilation=1/2/4 的三路卷积，每路输出 1 通道；拼接后的 3 通道表示三个尺度的模式响应
        #---------------------------------------------------------------------------------
        # 当前分支：[B,64,H,W] -> [B,3,H,W]；历史分支：[B,3,D,H,W] -> [B,3,D,H,W]
        # 历史分支输出的 3 通道此时表示尺度，已不是历史帧维度；下方变量被重新赋值为这些模式特征
        # 实现差异：当前分支为 2D 卷积，历史分支为 3D 卷积 + GELU + GroupNorm，并非完全对称的两路
        # 当前 depth=1：三个尺度拼接后统一计算一次余弦亲和度，不是分别产生三张亲和度图再拼接
        curr_feature = self.curr_patch(curr_feature) # 把当前帧的 64 通道特征，转换成三个空间尺度的局部模式响应，供 CPA 与历史特征比较。
        batch_waped_feature = self.warped_patch(batch_waped_feature) # 从对齐后的历史特征体中提取三个尺度的局部模式，供 CPA 与当前帧比较。 前面ADR生成“待融合的历史内容”，warped_patch 生成“用于比较的历史模式”。

        # 两路沿 3 个模式通道去均值；当前模式沿 D 维复制为 [B,3,D,H,W]，与历史模式逐位置比较
        # 沿模式通道计算余弦相似度得到 [B,D,H,W]，补回通道维得到亲和度 [B,1,D,H,W]
        # 该权重未经 softmax/sigmoid，范围约为 [-1,1]，不是深度概率；负权重可以翻转特征符号
        temporal_volume = self.cossim((curr_feature-curr_feature.mean(1).unsqueeze(1)).unsqueeze(2).repeat(1,1,self.D,1,1), (batch_waped_feature-batch_waped_feature.mean(1).unsqueeze(1))).unsqueeze(1)
        # [B,1,D,H,W] 广播乘以 ADR 内容 [B,3,D,H,W]；同一位置的一个亲和度共同调制 3 个内容通道
        # 当前特征通过权重影响结果，不直接拼接到时序体；被加权的内容来自历史分支
        temporal_volume = (temporal_volume) * defomable_batch_waped_feature # 使用 CPA 亲和度对 ADR 输出加权

        # *---------------------------------------------#
        # * 4.10 编码时序视锥体，并汇聚为时序体素特征
        # 输入是 CPA 权重调制后的 ADR 内容 [B,3,112,48,160]，仍处于深度假设与图像坐标组成的视锥网格
        # temporal_prehourglass 在本类 __init__ 中定义，由前置卷积、hourglass(32) 和后置卷积共同组成
        # 通道流程：3 -> 前置 3D 卷积扩展到 32 -> Hourglass(32) -> 后置 3D 卷积映射到 64
        # Hourglass（沙漏网络）实现位于同目录 gwc_encoder.py：先逐级下采样，再上采样并结合跳跃连接
        # 下采样扩大有效感受野，聚合更大范围的深度和空间上下文；上采样恢复分辨率，跳跃连接补充局部信息
        # 它继续整合加权后的时序内容，不负责估计相机位姿或计算 CPA 权重，也尚未将特征转换到 XYZ 网格
        # 整个模块 [B,3,112,48,160] -> [B,64,112,48,160]；3 到 64 的变化不是 Hourglass 单独完成的
        temporal_volume = self.temporal_prehourglass(temporal_volume)
        # 整理为 [B,N,D,H,W,64]，复用主分支 geom 汇聚到 XYZ 网格，得到 [B,64,128,128,16]；当前 N=1
        temporal_volume = temporal_volume.view(B, N, -1, self.D, H, W)  # 当前 N=1：[B,64,112,48,160] -> [B,1,64,112,48,160]，-1 推断为通道数 64
        temporal_volume = temporal_volume.permute(0, 1, 3, 4, 5, 2)  # [B,1,64,112,48,160] -> [B,1,112,48,160,64]，通道维移至末尾：[B,N,C,D,H,W] -> [B,N,D,H,W,C]
        temporal_volume = self.voxel_pooling(geom, temporal_volume)  # geom=[B,1,112,48,160,3]；特征 [B,1,112,48,160,64] -> [B,64,128,128,16]，由视锥网格汇聚到 XYZ 网格
        # 进一步编码为 [B, 384, 128, 128, 16]，用单元素列表包装，供后续融合与占用监督使用
        temporal_volume =  self.temporal_hourglass(temporal_volume)  # [B,64,128,128,16] -> [B,384,128,128,16]，通道数 64 -> 384，输出 XYZ 网格尺寸不变
        temporal_voxel = [temporal_volume]
        # 外部 bev_encoder 将该时序体素与主分支融合；训练时它也单独经过共享占用头计算损失

        # *---------------------------------------------#
        # * 4.11 返回主分支体素、融合深度概率和时序体素
        # 对应调用处的 x、depth、temporal_voxel；主分支与时序体素的最终融合在外部 bev_encoder 中完成
        return bev_feat, depth_prob, temporal_voxel
