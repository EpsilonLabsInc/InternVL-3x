set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-32}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch


TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LR=1e-5
MAX_DYNAMIC_PATCH=6

prefix="./training/"

this_run="internvl3_8b_${TIMESTAMP}_${LR}_sf_1124_hand_length_15"

OUTPUT_DIR="${prefix}${this_run}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "pretrained/InternVL3-8B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/train_degen.json" \
 --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers 16 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_total_limit 3 \
  --learning_rate ${LR} \
  --weight_decay 0.001 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "wandb" \
  --wandb_project "vlm-hand-findings-len" \
  --wandb_run_name "${this_run}" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

# Upload checkpoint to R2 after training completes
TRAINING_EXIT_CODE=$?
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
  echo "Training completed successfully. Uploading checkpoint to R2..."
  rclone copy "${OUTPUT_DIR}" "r2:checkpoints/vlm/training/${this_run}" \
    --progress \
    --transfers 8 \
    --checkers 16 \
    --s3-chunk-size 50M
  echo "Checkpoint uploaded to R2: r2:checkpoints/vlm/training/${this_run}"
else
  echo "Training failed with exit code $TRAINING_EXIT_CODE. Skipping R2 upload."
fi