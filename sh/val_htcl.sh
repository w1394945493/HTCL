
PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
torchrun --nproc_per_node=2 \
    tools/test.py \
    projects/configs/occupancy/semantickitti/temporal_baseline_custom.py \
    /c20250502/wangyushen/Outputs/htcl/htcl/semkitti/train/epoch_1.pth \
    --launcher pytorch \
    --deterministic \
    --tmpdir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/htcl/val/.dist_test \
    --metrics-out /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/htcl/val/epoch_1_metrics.json \
    --eval mAP