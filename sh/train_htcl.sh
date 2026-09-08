#!/usr/bin/env bash

CHECKPOINT_DIR="/c20250502/wangyushen/Outputs/htcl/htcl/semkitti/train"
LATEST_PTH=$(ls -v "${CHECKPOINT_DIR}"/epoch_*.pth 2>/dev/null | tail -n 1)

echo "--------------------------------------------------"
echo "Resume checkpoint: ${LATEST_PTH:-not found, training from scratch}"
echo "--------------------------------------------------"

if [ -f "${LATEST_PTH}" ]; then
    RESUME_ARG="--resume-from ${LATEST_PTH}"
else
    RESUME_ARG=""
fi

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
python -m torch.distributed.launch \
    --nproc_per_node="${MLP_WORKER_GPU}" \
    --master_addr="${MLP_WORKER_0_HOST}" \
    --node_rank="${MLP_ROLE_INDEX}" \
    --master_port="${MLP_WORKER_0_PORT}" \
    --nnodes="${MLP_WORKER_NUM}" \
    tools/train.py \
    projects/configs/occupancy/semantickitti/temporal_baseline_custom.py \
    --launcher pytorch \
    --work-dir "${CHECKPOINT_DIR}" \
    ${RESUME_ARG}
