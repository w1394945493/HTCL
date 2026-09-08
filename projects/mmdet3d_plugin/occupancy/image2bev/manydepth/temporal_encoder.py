
import os
import json
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from collections import OrderedDict

import torch
from torchvision import transforms
import sys
sys.path.append("projects/mmdet3d_plugin/occupancy/image2bev/manydepth/")
import networks
from torch.autograd import Variable

# *=======================================================#
# * Aligned Temporal Volume Construction：基于位姿估计与单应性变换构建对齐时序体
class temporal_encoder(torch.nn.Module):
    def __init__(self, maxdisp, width, height  ):
        super(temporal_encoder, self).__init__()
        self.maxdisp = maxdisp # 112
        # * ==============================================#
        # * 定义轻量级 PoseNet 和基于深度假设平面的时序匹配编码器
        self.pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
        self.pose_dec = networks.PoseDecoder(self.pose_enc.num_ch_enc, num_input_features=1,
                                        num_frames_to_predict_for=2)
        self.encoder = networks.ResnetEncoderMatching(18, False,
                                                input_width = width,
                                                input_height = height,
                                                adaptive_bins=True,
                                                num_depth_bins=maxdisp )
        # * 冻结 PoseNet 参数；训练时仅将其用于推断帧间相对位姿
        for name, p in self.named_parameters():
            if name.startswith("pose"):
                p.requires_grad = False


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
        B, T, C, H, W = source_images.shape # (1 3 3 384 1280)
        combined_waped_feature = torch.zeros( B, T, self.maxdisp, H//4, W//4 ).cuda() # (1 3 112 96 320)

        height, width = ref_images.shape[-2: ] # 384 1280

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
        ref_image = ref_images.squeeze(1)
        for temporal in range(0, T):
            source_image = source_images[:, temporal, ...]
            input_image, original_size = self.load_and_preprocess_image(ref_image )
            source_image, _ = self.load_and_preprocess_image(source_image )
            # * 使用 PoseNet 根据当前图像与历史图像估计相对位姿；相机内参用于后续几何 warp
            with torch.no_grad():
            # Estimate poses
                pose_inputs = [source_image, input_image]
                pose_inputs = self.pose_enc(torch.cat(pose_inputs, 1))
                pose_inputs = [ pose_inputs ]
                axisangle, translation = self.pose_dec(pose_inputs) # * 输出相对旋转的轴角表示与相对平移
                pose = networks.transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)

            # * Homography warping / feature matching：基于相对位姿和相机内参，将历史特征对齐到当前帧
            curr_feature, batch_waped_feature  = self.encoder(current_image=input_image, # * 当前图像
                                            lookup_images=source_image.unsqueeze(1),     # * 历史图像
                                            poses=pose.unsqueeze(1),                     # * 相对位姿
                                            K=K,                                         # * 相机内参
                                            invK=invK)                                   # * 逆内参
            combined_waped_feature[:, temporal,:,:,:] = batch_waped_feature.squeeze(1)

        return  curr_feature, combined_waped_feature
