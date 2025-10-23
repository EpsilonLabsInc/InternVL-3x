import gc
import json
import os
import pickle
import sys
import time
from io import BytesIO

import numpy as np
import pydicom
import torch
import torchvision.transforms as T
from internvl.model.internvl_chat import InternVLChatModel
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoTokenizer


# Remove all distributed logic (no DDP, no init_distributed)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dcm_2_rgb(dcm_data, image_path):
    if hasattr(dcm_data, "pixel_array"):
        pixel_array = dcm_data.pixel_array
    else:
        print("111", image_path)

    pixel_array_normalized = (
        (pixel_array - np.min(pixel_array))
        / (np.max(pixel_array) - np.min(pixel_array))
        * 255
    )
    pixel_array_normalized = pixel_array_normalized.astype(np.uint8)
    rgb_array = np.stack([pixel_array_normalized] * 3, axis=-1)
    rgb_image = Image.fromarray(rgb_array)

    rows = dcm_data.Rows
    cols = dcm_data.Columns
    if rows * cols > 16000000:
        new_size = (cols // 2, rows // 2)
        rgb_image = rgb_image.resize(new_size, Image.Resampling.LANCZOS)

    return rgb_image


generation_config = dict(
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.5,
    top_k=100,
    num_beams=2,
    repetition_penalty=1.5,
)

print("generation_config: ", generation_config)


def get_dcm_from_local(local_path):
    dicom_file = pydicom.dcmread(local_path)
    return dicom_file


def load_image(image_file, input_size=448, max_num=12):
    if "dcm" in image_file:
        dcm_data = get_dcm_from_local(image_file)
        image = dcm_2_rgb(dcm_data, image_file)
    else:
        image = Image.open(image_file).convert("RGB")

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


@torch.inference_mode()
def generate_output(lines, model, tokenizer, output_path):
    results = []
    times = []

    for line in tqdm(lines, desc="Processing"):
        start_time = time.time()
        entry = json.loads(line)

        image_paths = entry["image"]
        entry["image"] = image_paths
        try:
            pixel_values_list = [
                load_image(image_path, max_num=12)
                    .to(torch.bfloat16)
                    .to(model.device)
                for image_path in image_paths
            ]
            pixel_values = torch.cat(pixel_values_list, dim=0)
            num_patches_list = [pv.size(0) for pv in pixel_values_list]

            query, truth_report = entry["conversations"]
            query = query["value"]
            truth_report = truth_report["value"]

            response = model.chat(
                tokenizer,
                pixel_values,
                query,
                generation_config,
                num_patches_list=num_patches_list,
            )

        except Exception as e:
            print(f"Error: {e}")
            print(entry)
            continue

        entry["truth"] = truth_report
        entry["generated"] = response
        results.append(entry)

        del pixel_values_list
        del pixel_values
        del response
        # force a cache “trim” (optional, only if you really need to see a lower nvidia-smi number)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        end_time = time.time()
        times.append(end_time - start_time)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)


def main():
    # Single‐GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    if len(sys.argv) < 2:
        print("Usage: python3 intern_evaluation_single_gpu.py <description>")
        sys.exit(1)

    description = sys.argv[1]
    test_jsonl = (
        "/home/eric/projects/InternVL-3x/output/jsonl/gradient/all_chest_0507_test.jsonl"
    )

    checkpoint_dir = (
        "/home/eric/projects/InternVL-3x/internvl_chat/training/"
        "internvl3_chimera_20250514_222713_1e-5_gradient_all_0513_continue"
    )

    output_dir = f"/home/eric/projects/InternVL-3x/internvl_chat/test_data/pkls/{description}"
    os.makedirs(output_dir, exist_ok=True)

    # Collect all checkpoints
    checkpoints = sorted(
        [
            os.path.join(checkpoint_dir, ckpt)
            for ckpt in os.listdir(checkpoint_dir)
            if ckpt.startswith("checkpoint-")
        ],
        key=lambda x: int(x.split("-")[-1]),
    )

    print(f"Found {len(checkpoints)} checkpoints to evaluate. They are:")
    for ckpt in checkpoints:
        print("  ", ckpt)

    # Load the entire dataset once
    with open(test_jsonl, "r") as f:
        all_lines = f.readlines()[:100]

    for checkpoint in checkpoints:
        suffix = checkpoint.split("/")[-1]
        print(f"\nLoading model from {checkpoint} (suffix: {suffix})")

        # Prepare output subfolder and path
        ckpt_output_dir = os.path.join(output_dir, suffix)
        os.makedirs(ckpt_output_dir, exist_ok=True)
        output_path = os.path.join(ckpt_output_dir, f"{suffix}.pkl")
        if os.path.exists(output_path):
            print(f"  → {output_path} already exists. Skipping checkpoint.")
            continue

        # Load model onto single GPU
        model = InternVLChatModel.from_pretrained(
            checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(device)

        model.config.output_hidden_states = False
        model.config.output_attentions    = False

        # Turn off past-key-values caching at generate-time:
        model.config.use_cache = False

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, use_fast=False
        )

        # Run inference over all lines
        generate_output(all_lines, model, tokenizer, output_path)

        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    print("\nAll checkpoints processed.")


if __name__ == "__main__":
    main()
