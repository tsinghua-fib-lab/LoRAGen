#!/usr/bin/env bash

# Generate per-task inference configs
python3 _generate_infer_configs.py

# Resolve base paths
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"
CONFIG_DIR="$BASE_DIR/configs/infer"
TARGET_JSON="$BASE_DIR/configs/infer/infer_target_tasks.json"

# Multi-GPU assignment (edit as needed)
GPU_LIST=(3 4)
GPU_COUNT=${#GPU_LIST[@]}

# Optional mirrors / local model paths
export HF_ENDPOINT=https://hf-mirror.com
export BERTSCORE_MODEL_DIR=LoRAGen/stage2/train/evaluation/eval_model/roberta-large

mkdir -p logs
mkdir -p logs/bash_infer

# Read task list from JSON
TARGETS=$(python3 -c "
import json
with open('$TARGET_JSON') as f:
    tasks = json.load(f)
print(' '.join(tasks))
")

# Locate the entrypoint
if [ -f "$BASE_DIR/main_stage2.py" ]; then
  TRAIN_SCRIPT="$BASE_DIR/main_stage2.py"
elif [ -f "$BASE_DIR/train/main_stage2.py" ]; then
  TRAIN_SCRIPT="$BASE_DIR/train/main_stage2.py"
else
  echo "ERROR: main_stage2.py not found (checked: $BASE_DIR/ and $BASE_DIR/train/)."
  exit 1
fi

# Launch tasks in parallel, one per selected GPU
i=0
for task in $TARGETS; do
  CONFIG_NAME="${task}.yaml"
  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}"
  GPU_ID=${GPU_LIST[$((i % GPU_COUNT))]}
  echo "Start: $CONFIG_PATH on GPU $GPU_ID"

  (
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    python "$TRAIN_SCRIPT" --config "$CONFIG_PATH" --mode infer > "logs/bash_infer/${CONFIG_NAME%.yaml}.log" 2>&1
    echo "Done:  $CONFIG_NAME on GPU $GPU_ID"
  ) &
  ((i++))
done

wait
echo "All inference jobs completed. See logs under logs/bash_infer/*.log"
