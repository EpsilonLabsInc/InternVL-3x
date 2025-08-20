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
  "192.168.0.113"  #5
  "192.168.0.211"  #6
  "192.168.0.137"  #7
  "192.168.0.110"  #8
  "192.168.0.66"   #9
  "192.168.0.136"  #10
  "192.168.0.170"  #11
  "192.168.0.229"  #12
  "192.168.0.228"  #13
  "192.168.0.48"   #14
  "192.168.0.6"    #15
  "192.168.0.93"   #16
  "192.168.0.68"   #17
)

# Local paths
json_file="all_0816_label_gpt_bp_only_findings.json"
script_file="internvl3_train_multinode.sh"
# label_files="/home/eric/projects/InternVL-3x/output/jsonl/all/all_08162025_labels_train_gpt_body_part.jsonl"
label_files = "/home/eric/projects/InternVL-3x/output/jsonl/all/all_08162025_labels_train_gpt_body_part_only_findings.jsonl"
# finetune_file="internvl/train/internvl_chat_finetune.py"

idx=1
for ip in "${vms[@]}"; do
  host="eric@${ip}"

  echo "=== [VM ${idx}/${#vms[@]} | ${ip}] Copying JSON file '${json_file}' → ${host}:/home/eric/projects/InternVL-3x/internvl_chat/shell/data/ ==="
  scp "${json_file}" \
      "${host}:/home/eric/projects/InternVL-3x/internvl_chat/shell/data/"
  echo "=== [VM ${idx}] Done copying '${json_file}' ==="
  echo

  echo "=== [VM ${idx}/${#vms[@]} | ${ip}] Copying training script '${script_file}' → ${host}:/home/eric/projects/InternVL-3x/internvl_chat/ ==="
  scp "${script_file}" \
      "${host}:/home/eric/projects/InternVL-3x/internvl_chat/"
  echo "=== [VM ${idx}] Done copying '${script_file}' ==="
  echo

  echo "=== [VM ${idx}/${#vms[@]} | ${ip}] Copying label file(s) '${label_files}' → ${host}:/home/eric/projects/InternVL-3x/output/jsonl/all/ ==="
  scp ${label_files} \
      "${host}:/home/eric/projects/InternVL-3x/output/jsonl/all/"
  echo "=== [VM ${idx}] Done copying '${label_files}' ==="
  echo

  # echo "=== [VM ${idx}/${#vms[@]} | ${ip}] Copying finetune file '${finetune_file}' → ${host}:/home/eric/projects/InternVL-3x/internvl_chat/internvl/train/ ==="
  # scp "${finetune_file}" \
  #     "${host}:/home/eric/projects/InternVL-3x/internvl_chat/internvl/train/"
  # echo "=== [VM ${idx}] Done copying '${finetune_file}' ==="
  # echo

  ((idx++))
done
