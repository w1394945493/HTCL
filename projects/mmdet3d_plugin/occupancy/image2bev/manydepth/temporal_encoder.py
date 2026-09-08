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
        B, T, C, H, W = source_images.shape # (1 3 3 384 1280) 历史帧图像(前三帧)
        combined_waped_feature = torch.zeros( B, T, self.maxdisp, H//4, W//4 ).cuda() # (1 3 112 96 320)

        height, width = ref_images.shape[-2: ] # 384 1280 当前帧图像

        # 注释原代码
        # intrinsics =  intrinsics.squeeze(1).cpu().detach().numpy() # (1 4 4)
        # K, invK = torch.zeros_like( torch.tensor(intrinsics)).cuda() , torch.zeros_like( torch.tensor(intrinsics)).cuda() # (1 4 4) (1 4 4)
        # for batch in range( 0, B ):
        #     K_, invK_ = self.load_and_preprocess_intrinsics(intrinsics[batch], width, height)
        #     K_ = Variable(K_, requires_grad=True).cuda()
        #     invK_ = Variable(invK_, requires_grad=True).cuda()
        #     K[batch], invK[batch] = K_, invK_

        # 使用纯 PyTorch 批量处理内参，避免 GPU→CPU→NumPy→GPU 的数据搬运和逐样本循环
        # 去除相机维度并切断无须保留的内参梯度 # 与原先 torch.Tensor(...) 的数据类型保持一致，并满足 pinv 的计算要求
        K = intrinsics.squeeze(1).detach().to(device=ref_images.device, dtype=torch.float32).clone()  # 创建独立副本，避免下方原地缩放修改输入 intrinsics
        K[:, 0, :] *= width // 4  # 将归一化内参的水平方向参数缩放到 1/4 尺度特征图
        K[:, 1, :] *= height // 4  # 将归一化内参的垂直方向参数缩放到 1/4 尺度特征图
        invK = torch.linalg.pinv(K)  # 批量计算每个样本的内参伪逆，用于几何反投影

        # *==============================================#
        # * 对应论文第 3.2 节：将当前帧分别与每个历史帧组成图像对
        ref_image = ref_images.squeeze(1) # (1 3 384 1280)
        for temporal in range(0, T): # T:3
            source_image = source_images[:, temporal, ...] # (1 3 384 1280)
            input_image, original_size = self.load_and_preprocess_image(ref_image )
            source_image, _ = self.load_and_preprocess_image(source_image )

            # *===========================================#
            # * 使用 PoseNet 根据当前图像与历史图像估计相对位姿；相机内参用于后续几何 warp
            # * (1) 将当前帧和历史帧输入轻量级 PoseNet，以估计用于光度重投影的相对相机位姿；
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
            # * Homography warping / feature matching：基于相对位姿和相机内参，将历史特征对齐到当前帧
            # * 使用ManyDepth风格的resnet18，在1/4尺度的二维图像特征图上，利用相机几何约束进行跨帧特征匹配
            # * (2) 生成当前帧特征图以及历史帧特征图集合：
            # * (3) 利用相对相机位姿和一组候选深度假设平面，通过单应性变换构建经过变换的历史帧特征
            curr_feature, batch_waped_feature  = self.encoder(current_image=input_image, # (1 3 384 1280) # * 当前图像
                                            lookup_images=source_image.unsqueeze(1),     # (1 1 3 384 1280) # * 历史图像
                                            poses=pose.unsqueeze(1),                     # (1 1 4 4) * 相对位姿
                                            K=K,                                         # (1 4 4) * 相机内参
                                            invK=invK)                                   # (1 4 4) * 逆内参
            combined_waped_feature[:, temporal,:,:,:] = batch_waped_feature.squeeze(1) # (1 3 112 96 320)

        return  curr_feature, combined_waped_feature # (1 64 96 320) (1 3 112 96 320)
