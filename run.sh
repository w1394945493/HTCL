pip install open3d -i https://pypi.org/simple
pip install timm -i https://pypi.org/simple
pip install PyMCubes -i https://pypi.org/simple

pip install --no-deps \
  /c20250502/wangyushen/whl/open3d-0.19.0-cp310-cp310-manylinux_2_31_x86_64.whl

pip install --no-deps "monai==0.9.1" \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --timeout 600

cd /vepfs-mlp2/c20250502/haoce/wangyushen/HTCL/projects/mmdet3d_plugin/occupancy/image2bev/dcn
python setup.py build_ext --inplace

# 制备PoseNet部分权重
python scripts/extract_posenet_weights.py \
    /c20250502/wangyushen/Weights/htcl/pretrain.pth \
    /c20250502/wangyushen/Weights/htcl/posenet.pth
  
# PYTHONPATH="$(pwd)"：会直接覆盖原来的 PYTHONPATH，最终只包含当前目录。
# PYTHONPATH="$(pwd):${PYTHONPATH}"：是在原有路径前面添加当前目录
# * semkitti训练
PYTHONPATH="$(pwd)" \
python tools/train.py   \
    projects/configs/occupancy/semantickitti/temporal_baseline_custom.py \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/htcl/train

# * semkitti评估
PYTHONPATH="$(pwd)" \
python /vepfs-mlp2/c20250502/haoce/wangyushen/HTCL/tools/test.py \
  /vepfs-mlp2/c20250502/haoce/wangyushen/HTCL/projects/configs/occupancy/semantickitti/temporal_baseline_custom.py \
  /c20250502/wangyushen/Weights/htcl/pretrain.pth \
  --out /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/htcl/val/ \
  --eval mAP

# * 多卡训练
export CUDA_VISIBLE_DEVICES=0,1
PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
torchrun --nproc_per_node=2 tools/train.py \
    projects/configs/occupancy/semantickitti/temporal_baseline_custom.py \
    --launcher pytorch \ 
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/htcl/train

# 火山服务器
cd /vepfs-mlp2/c20250502/haoce/wangyushen/HTCL
. /root/miniconda3/bin/activate
conda activate /vepfs-mlp2/c20250502/haoce/conda_env/wys_temp_2
bash sh/train_htcl.sh




