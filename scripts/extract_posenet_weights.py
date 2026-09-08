#!/usr/bin/env python3
"""从 HTCL 完整 checkpoint 中提取 PoseNet 权重。"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


POSE_PREFIX = "img_view_transformer.temporal_encoder."


def parse_args():
    parser = argparse.ArgumentParser(description="提取 HTCL checkpoint 中的 PoseNet 权重")
    parser.add_argument("input", type=Path, help="输入的完整 .pth checkpoint")
    parser.add_argument("output", type=Path, help="输出的 PoseNet .pth 文件")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    pose_state_dict = OrderedDict()
    for key, value in state_dict.items():
        normalized_key = key.removeprefix("module.")
        if not normalized_key.startswith(POSE_PREFIX):
            continue

        relative_key = normalized_key[len(POSE_PREFIX):]
        if relative_key.startswith(("pose_enc.", "pose_dec.")):
            pose_state_dict[relative_key] = value

    if not pose_state_dict:
        raise RuntimeError(f"{args.input} 中没有找到 PoseNet 权重")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": pose_state_dict}, args.output)
    print(f"已提取 {len(pose_state_dict)} 个 PoseNet 参数张量")
    print(f"保存位置：{args.output}")


if __name__ == "__main__":
    main()
