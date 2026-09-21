# import os
# import json
# import argparse
# import numpy as np
# import PIL.Image as pil
# import matplotlib as mpl
# import matplotlib.cm as cm
# from collections import OrderedDict

import torch
# from torchvision import transforms
from torch.autograd import Variable
from .networks import ResnetEncoder,PoseDecoder,ResnetEncoderMatching,transformation_from_parameters

# *=======================================================#
# * Aligned Temporal Volume Construction：基于位姿估计与单应性变换构建对齐时序体
class temporal_encoder(torch.nn.Module):
    def __init__(self, maxdisp, width, height, pose_pretrained=None):
        super(temporal_encoder, self).__init__()
        self.maxdisp = maxdisp # 112
        # * ==============================================#
        # * 定义轻量级 PoseNet 和基于深度假设平面的时序匹配编码器
        self.pose_enc = ResnetEncoder(18, False, num_input_images=2)
        self.pose_dec = PoseDecoder(self.pose_enc.num_ch_enc, num_input_features=1,
                                        num_frames_to_predict_for=2)
        self.encoder = ResnetEncoderMatching(18, False,
                                                input_width = width,
                                                input_height = height,
                                                min_depth_bin=0,
                                                max_depth_bin=maxdisp,
                                                adaptive_bins=False,
                                                num_depth_bins=maxdisp )
        # * 单独加载预训练 PoseNet；文件中的参数键应为 pose_enc.* 和 pose_dec.*
        if pose_pretrained is not None:
            self.load_pose_pretrained(pose_pretrained)

        # * 冻结 PoseNet 参数；训练时仅将其用于推断帧间相对位姿
        for name, p in self.named_parameters():
            if name.startswith("pose"):
                p.requires_grad = False

    def load_pose_pretrained(self, checkpoint_path):
        """加载由 scripts/extract_posenet_weights.py 提取的 PoseNet 权重。"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        pose_state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
        incompatible = self.load_state_dict(pose_state_dict, strict=False)
        missing_pose_keys = [
            key for key in incompatible.missing_keys
            if key.startswith(("pose_enc.", "pose_dec."))
        ]
        if missing_pose_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "PoseNet 权重不完整或参数名称不匹配："
                f"missing={missing_pose_keys}, unexpected={incompatible.unexpected_keys}"
            )
        print(f"Loaded PoseNet checkpoint from {checkpoint_path}")

    def load_and_preprocess_image(self, image ):

        batch, channel, original_width, original_height = image.shape
        image = image.div(255)
        image = Variable(image, requires_grad=True).cuda()
        if torch.cuda.is_available():
            return image, (original_height, original_width)
        return image, (original_height, original_width)

    # 将一张图像的归一化相机内参转换为 1/4 特征图尺度下的相机内参 K，并计算它的伪逆 invK。
    # def load_and_preprocess_intrinsics(self, intrinsics, resize_width, resize_height):
    #     K = np.array( intrinsics )
    #     K[0, :] *= resize_width // 4
    #     K[1, :] *= resize_height // 4
    #     invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
    #     invK = Variable(invK, requires_grad=True).cuda()
    #     K = torch.Tensor(K).unsqueeze(0)
    #     K = Variable(K, requires_grad=True).cuda()
    #     return K , invK

    def forward(self, ref_images, source_images, intrinsics, calib=None ):
        # *==============================================#
        # * 4.7.1 准备参考帧、历史帧与时序输出容器
        # ref 是 reference，当前帧提供对齐基准；source 是历史来源帧，提供待采样的特征
        # 此处 T 仅表示历史帧数，当前为 3；与外部包含当前帧的队列长度 4 不同
        # self.maxdisp 在此表示深度假设数 D=112，不是输出视差值；calib 参数在本函数中未使用
        B, T, C, H, W = source_images.shape # (1 3 3 384 1280) 历史帧图像(前三帧)
        combined_waped_feature = torch.zeros( B, T, self.maxdisp, H//4, W//4 ).cuda() # (1 3 112 96 320)

        height, width = ref_images.shape[-2: ] # 384 1280 当前帧图像

        # *==============================================#
        # * 4.7.2 准备特征图尺度的相机内参与逆内参
        # 下方缩放以输入为归一化内参为前提；K 用于投影，invK 用于从当前像素反投影
        # 保留的旧实现：通过 NumPy 逐样本处理内参，当前不执行
        # intrinsics =  intrinsics.squeeze(1).cpu().detach().numpy() # (1 4 4)
        # K, invK = torch.zeros_like( torch.tensor(intrinsics)).cuda() , torch.zeros_like( torch.tensor(intrinsics)).cuda() # (1 4 4) (1 4 4)
        # for batch in range( 0, B ):
        #     K_, invK_ = self.load_and_preprocess_intrinsics(intrinsics[batch], width, height)
        #     K_ = Variable(K_, requires_grad=True).cuda()
        #     invK_ = Variable(invK_, requires_grad=True).cuda()
        #     K[batch], invK[batch] = K_, invK_

        # 使用纯 PyTorch 批量处理内参，避免 GPU→CPU→NumPy→GPU 的数据搬运和逐样本循环
        # 去除相机维度，切断内参梯度，并转为 float32 以满足 pinv 的计算要求
        K = intrinsics.squeeze(1).detach().to(device=ref_images.device, dtype=torch.float32).clone()  # 创建独立副本，避免下方原地缩放修改输入 intrinsics
        K[:, 0, :] *= width // 4  # 将归一化内参的水平方向参数缩放到 1/4 尺度特征图
        K[:, 1, :] *= height // 4  # 将归一化内参的垂直方向参数缩放到 1/4 尺度特征图
        invK = torch.linalg.pinv(K)  # 批量计算每个样本的内参伪逆，用于几何反投影

        # *==============================================#
        # * 4.7.3 逐张遍历历史帧，组成“历史帧 + 当前帧”图像对
        # 每对图像分别除以 255 归一化；下面的位姿估计、特征对齐与结果写入对每张历史图执行一次
        ref_image = ref_images.squeeze(1) # (1 3 384 1280)
        for temporal in range(0, T): # T:3
            source_image = source_images[:, temporal, ...] # (1 3 384 1280)
            input_image, original_size = self.load_and_preprocess_image(ref_image )
            source_image, _ = self.load_and_preprocess_image(source_image )

            # *===========================================#
            # * 4.7.4 使用冻结的 PoseNet 估计当前帧与该历史帧的相对位姿
            # 图像对 -> 位姿编码器 -> 旋转轴角与平移 -> [B,4,4] 变换矩阵；内参用于后续几何对齐
            with torch.no_grad():  # 位姿网络仅用于推理，不构建计算图，也不更新 PoseNet 参数
                pose_inputs = [source_image, input_image]  # 按“历史帧、当前帧”的顺序组成待估计位姿的图像对
                pose_inputs = torch.cat(pose_inputs, 1)  # 沿通道维拼接两帧图像：(B, 3, H, W)×2 → (B, 6, H, W)
                pose_inputs = self.pose_enc(pose_inputs)  # 使用 PoseNet 编码器提取图像对的多尺度位姿特征
                pose_inputs = [pose_inputs]  # 按 PoseDecoder 要求，将编码器的多尺度特征包装为输入列表
                axisangle, translation = self.pose_dec(pose_inputs)  # 解码相对旋转轴角和平移，形状均为 (B, 2, 1, 3)
                pose = transformation_from_parameters(  # 将轴角和平移转换为齐次相对变换矩阵 (B, 4, 4)
                    axisangle[:, 0],  # 选择第一个待预测帧对应的旋转参数，形状为 (B, 1, 3)
                    translation[:, 0],  # 选择第一个待预测帧对应的平移参数，形状为 (B, 1, 3)
                    invert=True,  # 对变换求逆，使矩阵方向满足后续历史帧特征对齐所需的坐标变换方向
                )

            # *===========================================#
            # * 4.7.5 提取二维特征，并按候选深度将历史特征对齐到当前帧
            # 实现：networks/resnet_encoder.py 的 ResnetEncoderMatching.forward() 与 match_features()
            # 独立于 PoseNet 的 ResNet18 提取 1/4 尺度特征；当前帧特征 curr_feature=[B,64,96,320]
            # 对每个候选深度：当前像素反投影 -> 相对位姿变换 -> 投影到历史图 -> grid_sample 采样
            # 历史对齐特征 [B,D,64,96,320] 沿 64 通道取均值，返回 batch_waped_feature=[B,D,96,320]
            # D=112 索引深度假设，元素值是采样特征的均值，不是深度值、深度概率或匹配相似度
            curr_feature, batch_waped_feature  = self.encoder(current_image=input_image, # (1 3 384 1280)，当前图像
                                            lookup_images=source_image.unsqueeze(1),     # (1 1 3 384 1280)，单张历史图像
                                            poses=pose.unsqueeze(1),                     # (1 1 4 4)，相对位姿
                                            K=K,                                         # (1 4 4)，相机内参
                                            invK=invK)                                   # (1 4 4)，内参伪逆
            # *----------------------------------------------#
            # * 4.7.6 将该历史帧的对齐结果写入时序容器
            # 写入第 temporal 个历史帧槽位，全部历史帧处理后为 [B,T,D,96,320]
            # 当前 D=112，batch_waped_feature 的第 1 维不是单元素维，因此 squeeze(1) 不改变形状
            combined_waped_feature[:, temporal,:,:,:] = batch_waped_feature.squeeze(1) # (1 3 112 96 320)

        # *==============================================#
        # * 4.7.7 返回当前帧特征与各历史帧的深度假设对齐特征体
        # curr_feature 为最后一次循环提取的当前帧特征；combined_waped_feature 保留全部 T 张历史帧的结果
        # 输出尚未汇聚为 XYZ 体素，后续由调用方进行 ADR 细化、CPA 加权及体素构建
        return  curr_feature, combined_waped_feature # (1 64 96 320) (1 3 112 96 320)
