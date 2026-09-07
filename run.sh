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


export PYTHONPATH="." 
python tools/train.py   \
    projects/configs/occupancy/semantickitti/temporal_baseline_custom.py \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/htcl/train

PYTHONPATH="$(pwd)" \
python tools/train.py   \
    projects/configs/occupancy/semantickitti/temporal_baseline_custom.py \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/htcl/train

export PYTHONPATH="."  
python /vepfs-mlp2/c20250502/haoce/wangyushen/HTCL/tools/test.py \
  /vepfs-mlp2/c20250502/haoce/wangyushen/HTCL/projects/configs/occupancy/semantickitti/temporal_baseline_custom.py \
  /c20250502/wangyushen/Weights/htcl/pretrain.pth \
  --out /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/htcl/val/ \
  --eval mAP