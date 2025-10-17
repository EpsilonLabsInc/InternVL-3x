#!/usr/bin/env bash
#
# transfer_to_vms.sh
# ------------------
# Copies `all_0802_nolabel.json`, `internvl3_train_multinode.sh`, and
# all `all_08022025_no_labels_train.jsonl` files to each VM, printing progress/errors to stdout.

# List of VM IPs
vms=(
  "192.168.0.0"    #1
  "192.168.0.107"  #2
  "192.168.0.187"  #3
  "192.168.0.70"   #4
  "192.168.0.233"  #5
  "192.168.0.211"  #6
  "192.168.0.236"  #7
  "192.168.0.110"  #8
  "192.168.0.66"   #9
  "192.168.0.190"  #10
  "192.168.0.210"  #11
  "192.168.0.229"  #12
  "192.168.0.228"  #13
  "192.168.0.48"   #14
  "192.168.0.6"    #15
  "192.168.0.85"   #16
  "192.168.0.68"   #17
  "192.168.0.69"   #18
  "192.168.0.129"  #19
  "192.168.0.212"  #20
  "192.168.0.100"  #21
  "192.168.0.202"  #22
  "192.168.0.50"   #23
  "192.168.0.1"    #24
  "192.168.0.206"  #25
  "192.168.0.72"   #26
  "192.168.0.71"   #27
  "192.168.0.7"    #28
  "192.168.0.92"   #29
  "192.168.0.97"   #30
)

# Local paths
# label_files="/home/eric/projects/InternVL-3x/output/jsonl/all/all_09042025_train.jsonl"
# label_files="/home/eric/projects/InternVL-3x/output/jsonl/all/all_09242025_train.jsonl"
label_files="/home/eric/projects/InternVL-3x/output/jsonl/all/1007_train_99.jsonl"
json_file="/home/eric/projects/InternVL-3x/internvl_chat/shell/data/train_1007_label_gpt.json"
script_file="internvl3_train_multinode_stage1.sh"

# finetune_file="internvl/train/internvl_chat_finetune.py"
# pretrained_folder="/home/eric/projects/InternVL-3x/internvl_chat/pretrained/InternVL3-chimera-38B-2B"
# tokenizer_files=(
#   "tokenizer_config.json"
#   "special_tokens_map.json"
#   "vocab.json"
#   "merges.txt"
#   "added_tokens.json"
# )

idx=1
for ip in "${vms[@]}"; do
  host="eric@${ip}"

  ################
  echo "=== [VM ${idx}/${#vms[@]} | ${ip}] Copying label file(s) '${label_files}' → ${host}:/home/eric/projects/InternVL-3x/output/jsonl/all/ ==="
  scp ${label_files} \
      "${host}:/home/eric/projects/InternVL-3x/output/jsonl/all/"
  echo "=== [VM ${idx}] Done copying '${label_files}' ==="
  echo

  ################
  echo "=== [VM ${idx}/${#vms[@]} | ${ip}] Copying JSON file '${json_file}' → ${host}:/home/eric/projects/InternVL-3x/internvl_chat/shell/data/ ==="
  scp "${json_file}" \
      "${host}:/home/eric/projects/InternVL-3x/internvl_chat/shell/data/"
  echo "=== [VM ${idx}] Done copying '${json_file}' ==="
  echo

  ################
  echo "=== [VM ${idx}/${#vms[@]} | ${ip}] Copying training script '${script_file}' → ${host}:/home/eric/projects/InternVL-3x/internvl_chat/ ==="
  scp "${script_file}" \
      "${host}:/home/eric/projects/InternVL-3x/internvl_chat/"
  echo "=== [VM ${idx}] Done copying '${script_file}' ==="
  echo

  # echo "=== [VM ${idx}/${#vms[@]} | ${ip}] Copying finetune file '${finetune_file}' → ${host}:/home/eric/projects/InternVL-3x/internvl_chat/internvl/train/ ==="
  # scp "${finetune_file}" \
  #     "${host}:/home/eric/projects/InternVL-3x/internvl_chat/internvl/train/"
  # echo "=== [VM ${idx}] Done copying '${finetune_file}' ==="
  # echo

  # echo "=== [VM ${idx}/${#vms[@]} | ${ip}] Copying pretrained folder '${pretrained_folder}' → ${host}:/home/eric/projects/InternVL-3x/internvl_chat/pretrained/ ==="
  # scp -r "${pretrained_folder}" \
  #     "${host}:/home/eric/projects/InternVL-3x/internvl_chat/pretrained/"
  # echo "=== [VM ${idx}] Done copying pretrained folder ==="
  # echo

  # echo "=== [VM ${idx}/${#vms[@]} | ${ip}] Creating pretrained directory and copying tokenizer files ==="
  # ssh "${host}" "mkdir -p /home/eric/projects/InternVL-3x/internvl_chat/pretrained/InternVL3-chimera-38B-2B"

  # for file in "${tokenizer_files[@]}"; do
  #   echo "  → Copying ${file}"
  #   echo "  → Full path: ${pretrained_folder}/${file}"  # Debug line
  #   scp "${pretrained_folder}/${file}" \
  #       "${host}:/home/eric/projects/InternVL-3x/internvl_chat/pretrained/InternVL3-chimera-38B-2B/"
  # done
  # echo "=== [VM ${idx}] Done copying tokenizer files ==="
  # echo


  ((idx++))
done
