set -x

GPUS=${GPUS:-2}
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

prefix="/home/eric/projects/InternVL-3x/internvl_chat/training/"

this_run="internvl3_chimera_${TIMESTAMP}_${LR}_mimic2_interview"

this_run="internvl3_chimera_${TIMESTAMP}_${LR}_gradient_all_0513_continue"

# this_run="internvl3_chimera_${TIMESTAMP}_${LR}_gradient_all_chest_0507"

OUTPUT_DIR="${prefix}${this_run}"


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 2
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 16
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/home/eric/projects/InternVL-3x/internvl_chat/training/internvl3_chimera_20250501_162810_1e-5_gradient_all_0501/checkpoint-67174" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/gradient_all_0428.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 36 \
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
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "wandb" \
  --wandb_project "internvl3_chimera_gradient_all_continue" \
  --wandb_run_name "${this_run}" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
