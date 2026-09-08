# import os
import numpy as np
# from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
# from typing import Type, Any, Callable, Union, List, Optional
from .temporal_retrieve  import *


class ResnetEncoderMatching(nn.Module):
    """Resnet encoder adapted to include a cost volume after the 2nd block.

    Setting adaptive_bins=True will recompute the depth bins used for matching upon each
    forward pass - this is required for training from monocular video as there is an unknown scale.
    """

    def __init__(self, num_layers, pretrained, input_height, input_width,
                 min_depth_bin=0.1, max_depth_bin=1, num_depth_bins=80,
                 adaptive_bins=False, depth_binning='linear'):

        super(ResnetEncoderMatching, self).__init__()

        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_depth_bins = num_depth_bins
        # we build the cost volume at 1/4 resolution
        self.matching_height, self.matching_width = input_height // 4, input_width // 4

        self.is_cuda = False

        resnets = {18: models.resnet18}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        encoder = resnets[num_layers](pretrained)
        self.layer0 = nn.Sequential(encoder.conv1,  encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool,  encoder.layer1)

        self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
                                              height=self.matching_height,
                                              width=self.matching_width)
        self.projector = Project3D(batch_size=self.num_depth_bins,
                                   height=self.matching_height,
                                   width=self.matching_width)

        # *====================================================#
        # * 新增代码
        # HTCL 使用固定深度范围，因此在初始化时生成一次深度假设，避免每次 forward 重复计算
        if self.depth_binning == 'linear':  # 在深度空间中均匀采样
            depth_bins = torch.linspace(
                min_depth_bin, max_depth_bin, steps=self.num_depth_bins,
                dtype=torch.float32)
        elif self.depth_binning == 'inverse':  # 在逆深度空间中均匀采样，并保持深度由近到远排列
            if min_depth_bin <= 0:
                raise ValueError("inverse depth binning requires min_depth_bin > 0")
            depth_bins = torch.reciprocal(torch.linspace(
                1.0 / min_depth_bin, 1.0 / max_depth_bin,
                steps=self.num_depth_bins, dtype=torch.float32))
        else:
            raise NotImplementedError(
                "Unsupported depth binning mode: {}".format(self.depth_binning))

        # buffer 不参与梯度更新，并会随模型自动迁移设备；persistent=False 避免增加 checkpoint 体积
        self.register_buffer("depth_bins", depth_bins, persistent=False)
        self.register_buffer("warp_depths", depth_bins[:, None, None, None].expand(-1, 1, self.matching_height, self.matching_width), persistent=False)

    #! 注释的原代码
    # def compute_depth_bins(self, min_depth_bin, max_depth_bin):
    #     """Compute the depths bins used to build the cost volume. Bins will depend upon
    #     self.depth_binning, to either be linear in depth (linear) or linear in inverse depth
    #     (inverse)"""

    #     if self.depth_binning == 'inverse':
    #         self.depth_bins = 1 / np.linspace(1 / max_depth_bin,
    #                                           1 / min_depth_bin,
    #                                           self.num_depth_bins)[::-1]  # maintain depth order

    #     elif self.depth_binning == 'linear':
    #         self.depth_bins = np.linspace(min_depth_bin, max_depth_bin, self.num_depth_bins)
    #     else:
    #         raise NotImplementedError
    #     self.depth_bins = torch.from_numpy(self.depth_bins).float()

    #     self.warp_depths = []
    #     for depth in self.depth_bins:
    #         depth = torch.ones((1, self.matching_height, self.matching_width)) * depth
    #         self.warp_depths.append(depth)
    #     self.warp_depths = torch.stack(self.warp_depths, 0).float()
    #     if self.is_cuda:
    #         self.warp_depths = self.warp_depths.cuda()

    def match_features(self, current_feats, lookup_feats, relative_poses, K, invK):
        """利用深度假设、相对位姿和相机内参，将历史帧特征对齐至当前帧视角。

        当前实现没有计算 current_feats 与 lookup_feats 的 L1 代价；current_feats 仅用于
        确定 batch 大小。外层目前每次只传入一张历史帧，因此历史帧维度 T 为 1。
        """

        batch_waped_feature = []  # 保存 batch 中每个样本经过几何对齐后的历史特征体

        # with torch.no_grad():  # 原代码曾考虑关闭 warp 分支梯度，当前保留梯度以参与反向传播
        for batch_idx in range(len(current_feats)):  # 逐个处理 batch 样本；current_feats 在此仅用于取得 B
            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1]  # (1 1 64 96 320) # 当前样本的历史特征：(1, T, C, h, w)
            _lookup_poses = relative_poses[batch_idx:batch_idx + 1]  # (1 1 4 4) # 当前样本各历史帧的相对位姿：(1, T, 4, 4)

            _K = K[batch_idx:batch_idx + 1]  # 当前样本在特征图尺度下的相机内参：(1, 4, 4)
            _invK = invK[batch_idx:batch_idx + 1]  # 当前样本的逆内参，用于像素反投影：(1, 4, 4)

            # * 反投影, 对应论文公式(3)
            # 将当前视角的像素按 D 个深度假设反投影为齐次三维点
            world_points = self.backprojector(self.warp_depths, _invK) # (112 4 30720)

            waped_feature = []  # 保存当前 batch 样本中每张历史帧的对齐结果
            for lookup_idx in range(_lookup_feats.shape[1]):  # 逐张处理当前样本包含的历史帧
                lookup_feat = _lookup_feats[:, lookup_idx]  # 取一张历史帧特征：(1, C, h, w)
                lookup_pose = _lookup_poses[:, lookup_idx]  # 取该历史帧的相对变换矩阵：(1, 4, 4)

                if lookup_pose.sum() == 0:  # 全零位姿表示对应历史帧缺失
                    continue  # 跳过缺失帧，不执行投影和特征采样
                # 为每个深度假设复制一份历史帧特征
                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1]) # (112 64 96 320) # (1, C, h, w) → (D, C, h, w)
                # 将三维假设点变换并投影到历史帧特征图
                pix_locs = self.projector(world_points, _K, lookup_pose)  # (112 96 320 2) # 输出 grid_sample 所需的坐标：(D, h, w, 2)
                warped = F.grid_sample(  # 根据投影坐标从历史帧特征图进行可微双线性采样
                    lookup_feat,
                    pix_locs,
                    padding_mode='zeros',  # 投影到特征图范围外的位置使用零填充
                    mode='bilinear',  # 对非整数采样位置执行双线性插值
                    align_corners=True,
                ) # (112 64 96 320) # 得到 D 个深度假设下的历史帧对齐特征：(D, C, h, w)
                waped_feature.append(warped)  # 保存当前历史帧的对齐特征

            # 将所有有效历史帧的对齐结果堆叠到时间维
            waped_feature = torch.stack(waped_feature, dim=0) # (1 112 64 96 320) # 形状为 (T, D, C, h, w)；当前调用中 T=1
            batch_waped_feature.append(waped_feature)  # 保存当前 batch 样本的对齐结果

        # 将所有 batch 样本的结果堆叠起来
        batch_waped_feature = torch.stack(batch_waped_feature, dim=0) # (1 1 112 64 96 320) # 形状为 (B, T, D, C, h, w)
        # 移除 T=1 的维度并交换 C、D
        batch_waped_feature = batch_waped_feature.squeeze(1).permute(0, 2, 1, 3, 4) # (1 64 112 96 320) # (B, 1, D, C, h, w) → (B, C, D, h, w)
        batch_waped_feature = batch_waped_feature.mean(1) # (1 112 96 320) # 沿特征通道求均值：(B, C, D, h, w) → (B, D, h, w)

        return batch_waped_feature  # 返回以深度假设为通道的历史帧对齐特征体

    def feature_extraction(self, image, return_all_feats=False):
        """ Run feature extraction on an image - first 2 blocks of ResNet"""

        image = (image - 0.45) / 0.225  # imagenet normalisation
        feats_0 = self.layer0(image)
        feats_1 = self.layer1(feats_0)

        if return_all_feats:
            return [feats_0, feats_1]
        else:
            return feats_1

    # def indices_to_disparity(self, indices):
    #     """Convert cost volume indices to 1/depth for visualisation"""
    #     batch, height, width = indices.shape
    #     depth = self.depth_bins[indices.reshape(-1).cpu()]
    #     disp = 1 / depth.reshape((batch, height, width))
    #     return disp

    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """ Returns a 'confidence' mask based on how many times a depth bin was observed"""

        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        confidence_mask = ((cost_volume > 0).sum(1) == num_bins_threshold).float()

        return confidence_mask

    def forward(self, current_image, lookup_images, poses, K, invK,
                min_depth_bin=0, max_depth_bin=112):  # 提取当前帧和历史帧特征，并构建对齐时序特征体
        # * (2) 生成当前帧特征图以及历史帧特征图集合：current_features 和 lookup_feats
        # * 提取当前帧特征
        # 使用共享的 ResNet 编码器提取当前帧多尺度特征
        # self.features = self.feature_extraction(current_image, return_all_feats=True)  # 返回所有尺度的特征，供当前分支及后续网络使用
        # current_feats = self.features[-1]  # 取最后一级特征作为帧间几何匹配的当前帧特征
        current_feats = self.feature_extraction(current_image, return_all_feats=False) # (1 64 96 320)

        # * 整理历史帧输入并生成用于几何匹配的深度假设平面
        #! 注释的原代码
        # with torch.no_grad():  # 本代码块仅包含深度分箱和形状变换，不构建对应的计算图
        #     if self.adaptive_bins:  # 启用自适应深度分箱时，为本次前向传播重新生成深度采样值
        #         # * 生成固定深度假设，不是需要学习的网络输出
        #         self.compute_depth_bins(min_depth_bin, max_depth_bin) # 在指定深度范围内生成 num_depth_bins 个深度假设平面
        #     batch_size, num_frames, chns, height, width = lookup_images.shape  # 解析历史图像形状：(B, T, C, H, W)
        #     # 合并 batch 维和时间维，以便一次送入二维图像编码器
        #     lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)  # (B, T, C, H, W) → (B×T, C, H, W)

        # * 新增代码
        batch_size, num_frames, chns, height, width = lookup_images.shape  # 解析历史图像形状：(B, T, C, H, W)
        # 合并 batch 维和时间维，以便批量提取历史帧特征
        lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)  # (B, T, C, H, W) → (B×T, C, H, W)

        # 使用与当前帧共享的编码器提取所有历史帧特征
        lookup_feats = self.feature_extraction(lookup_images, return_all_feats=False)  # (1 64 96 320) # 只返回几何匹配所需的最后一级特征
        _, chns, height, width = lookup_feats.shape  # 读取历史特征的通道数和空间尺寸
        # 恢复历史帧的 batch 维和时间维
        lookup_feats = lookup_feats.reshape(batch_size, num_frames, chns, height, width)  # (1 1 64 96 320) # (B×T, C, h, w) → (B, T, C, h, w)

        # *==============================================================#
        # * 基于相对位姿、相机内参和多深度假设，将历史特征反向采样到当前帧视角
        # * (3) 利用相对相机位姿和一组候选深度假设平面，通过单应性变换构建经过变换的历史帧特征
        batch_waped_feature = self.match_features(  # 对历史特征执行几何 warp，构建对齐后的时序特征体
            current_feats,  # (1 64 96 320) # 当前帧特征，用于确定 batch 和目标视角
            lookup_feats, # (1 1 64 96 320) # 尚未对齐的历史帧特征
            poses,  # (1 1 4 4) # 当前帧与各历史帧之间的相对位姿
            K,  # (1 4 4) # 特征图尺度下的相机内参矩阵
            invK,  # (1 4 4) # 相机内参伪逆，用于将二维像素反投影到三维空间
        )

        return current_feats, batch_waped_feature # (1 64 96 320) # (1 112 96 320) 返回当前帧特征及对齐后的历史时序特征体

    # def cuda(self):
    #     super().cuda()
    #     self.backprojector.cuda()
    #     self.projector.cuda()
    #     self.is_cuda = True
    #     if self.warp_depths is not None:
    #         self.warp_depths = self.warp_depths.cuda()

    # def cpu(self):
    #     super().cpu()
    #     self.backprojector.cpu()
    #     self.projector.cpu()
    #     self.is_cuda = False
    #     if self.warp_depths is not None:
    #         self.warp_depths = self.warp_depths.cpu()

    # def to(self, device):
    #     if str(device) == 'cpu':
    #         self.cpu()
    #     elif str(device) == 'cuda':
    #         self.cuda()
    #     else:
    #         raise NotImplementedError


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]

    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1, **kwargs):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        # if num_input_images > 1:
        #     self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        # else:
        #     self.encoder = resnets[num_layers](pretrained)

        #######################################
        self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False).cuda()

    def forward(self, depth, inv_K):

        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.cuda().view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot
