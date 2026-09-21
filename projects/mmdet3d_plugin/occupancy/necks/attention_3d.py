import torch
from torch import nn
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import matplotlib.animation as animation
from mmcv.cnn import build_conv_layer, build_norm_layer
norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv3d(query_dim, hidden_dim, 1, bias=False)
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)


    def forward(self, x):
        b, c, h, w, z = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv )
        q = q * self.scale
        sim = einsum("b h d i , b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y z) d -> b (h d) x y z", x=h, y=w)
        return self.to_out(out)

class LinearAttention3D(nn.Module) :
    # WVA 的注意力内容计算：主分支提供 query，时序体素提供 key/value；alpha 加权与残差相加在 neck 外层完成
    def __init__(self, dim,query_dim, heads=4, dim_head=2 ):
        super().__init__()
        # *==============================================#
        # * 5.2.4.1 初始化多头投影与输出映射
        # 当前 neck 设置 dim=query_dim=384、heads=2，dim_head 使用默认值 2，因此 hidden_dim=4
        # 1x1x1 卷积在每个体素位置混合通道，不改变 XYZ 网格尺寸；输出映射再将 4 通道恢复到 384
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_q = nn.Conv3d(query_dim, hidden_dim, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))
    def forward(self, x, query ):
        # return x
        # *----------------------------------------------#
        # * 5.2.4.2 从时序体素生成 key/value，并展开体素位置
        # x 为时序体素，query 为主分支体素；当前均为 [B,384,128,128,16]
        # 记 A=heads、F=dim_head、L=X*Y*Z；局部 h/w/z 是网格尺寸，rearrange 中 h 表示头数，勿混淆
        b, c, h, w, z = x.shape
        # [B,384,X,Y,Z] -> [B,3*A*F,X,Y,Z] -> 三个 [B,A*F,X,Y,Z]
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # 每个张量整理为 [B,A,F,L]；这里从时序体素得到的 q 随后被主分支 q 覆盖，不参与最终注意力
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv )

        # *----------------------------------------------#
        # * 5.2.4.3 从主分支生成真正的 query
        # [B,384,X,Y,Z] -> [B,A*F,X,Y,Z] -> [B,A,F,L]；当前为 [B,2,2,262144]
        query = self.to_q(query)
        q = rearrange(query, "b (h c) x y z -> b h c (x y z)", h=self.heads)
        # *----------------------------------------------#
        # * 5.2.4.4 分别归一化 query 的通道与 key 的体素位置
        # q 在每个头的 F 个通道上 softmax；k 在 L 个体素位置上 softmax；v 不做 softmax
        # 这不是标准 softmax(Q^T K)：两路分别归一化后，以不同乘法顺序实现线性注意力
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        # *----------------------------------------------#
        # * 5.2.4.5 沿全部时序体素位置汇总 key/value，形成紧凑上下文
        # context[b,a,d,e] = sum_n k[b,a,d,n] * v[b,a,e,n]，形状 [B,A,F,F]，当前 [B,2,2,2]
        # 每个 key 通道定义一组空间权重，对 value 做全局加权汇总；d/e 是头内通道，不是深度维
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        # *----------------------------------------------#
        # * 5.2.4.6 用各主分支体素的 query 读取上下文
        # out[b,a,e,n] = sum_d context[b,a,d,e] * q[b,a,d,n]，得到 [B,A,F,L]
        # 先汇总 K/V 再应用 Q，避免显式构造 [B,A,L,L]；固定头数和通道数时，核心计算随 L 线性增长
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        # *----------------------------------------------#
        # * 5.2.4.7 恢复 XYZ 网格，映射回体素通道数并返回
        # [B,A,F,L] -> [B,A*F,X,Y,Z]；使用 x 的网格尺寸还原，当前 query 与 x 的网格一致
        out = rearrange(out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=h, y=w)
        # 1x1x1 Conv3d + GroupNorm：[B,4,128,128,16] -> [B,384,128,128,16]
        # 返回的是检索出的时序内容；外部 SECONDFPN3D 再执行主分支 + alpha * 此输出
        return self.to_out(out)

def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=pad, bias=False),
                         build_norm_layer(norm_cfg, out_channels)[1] )
class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))
        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6

