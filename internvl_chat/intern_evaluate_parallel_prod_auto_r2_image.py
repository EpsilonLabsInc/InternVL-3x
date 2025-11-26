import gc
import json
import os
import pickle
import sys
import time
from io import BytesIO
import io
import logging

import numpy as np
import pydicom
import torch
import torch.distributed as dist
import torchvision.transforms as T
import boto3

# from google.cloud import storage
from internvl.model.internvl_chat import InternVLChatModel
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoTokenizer
import math


import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
os.environ["GOOGLE_CLOUD_DISABLE_GRPC"] = "true"

def init_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


sys.path.append("/root/projects/InternVL-3x/internvl_chat")
sys.path.append("/root/projects/epsutils")

from epsutils.dicom import dicom_utils, dicom_compression_utils
from epsutils.image import image_utils

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

logger = logging.getLogger(__name__)

# R2 S3 Configuration
_r2_client = None
R2_BUCKET_NAME = 'epsilonlabs-datasets'
R2_BUCKET_PNG_PREFIX = 'png/org-size'
# R2_BUCKET_PNG_PREFIX = 'png/prod/'
R2_STREAM_FROM_R2 = True

# Get R2 credentials from environment variables
R2_ENDPOINT_URL = os.getenv('R2_ENDPOINT_URL')
R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')


def get_r2_client():
    """Lazily initialize R2 client"""
    global _r2_client
    if _r2_client is None and R2_ENDPOINT_URL and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
        from botocore.config import Config

        config = Config(
            max_pool_connections=50,
            retries={'max_attempts': 5, 'mode': 'adaptive'},
            connect_timeout=30,
            read_timeout=60
        )

        _r2_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=config
        )
        logger.info(f'[Evaluation] R2 client initialized with endpoint: {R2_ENDPOINT_URL}, bucket: {R2_BUCKET_NAME}')
    return _r2_client


def _fetch_image_from_r2(filepath, max_retries=3):
    """Fetch image from Cloudflare R2 bucket with retry logic"""
    # Remove leading slash if present and prepend bucket PNG prefix if specified
    key = filepath.lstrip('/')
    if R2_BUCKET_PNG_PREFIX:
        key = f"{R2_BUCKET_PNG_PREFIX.rstrip('/')}/{key}"

    last_error = None
    r2_client = get_r2_client()

    for attempt in range(max_retries):
        try:
            if r2_client is None:
                raise ValueError("R2 client is not initialized")

            response = r2_client.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            body = response['Body']
            try:
                img_bytes = body.read()
            finally:
                body.close()
            return Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # exponential backoff: 2s, 4s, 6s
                logger.warning(f'Attempt {attempt + 1}/{max_retries} failed for {key}: {e}. Retrying in {wait_time}s...')
                time.sleep(wait_time)
            else:
                logger.error(f'SKIPPING SAMPLE: Failed to load {key} from R2 after {max_retries} attempts: {last_error}')

    # Raise exception to skip this sample
    raise FileNotFoundError(f'Failed to fetch image from R2: {key} after {max_retries} attempts. Last error: {last_error}')


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

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dcm_2_rgb(dcm_data, image_path, augmentation_parameters=None):
    image = dicom_utils.get_dicom_image_from_dataset(
        dcm_data, custom_windowing_parameters={"window_center": 0, "window_width": 0}
    )
    image = image_utils.numpy_array_to_pil_image(image, convert_to_rgb=True)

    rows = dcm_data.Rows
    cols = dcm_data.Columns
    # 1.6M pixels seems to cause issue of OOM during training
    while rows * cols > 14000000:
        # Compress the image by resizing by a factor of 2
        # rows = rows // 2
        # cols = cols // 2

        factor = math.sqrt(2)
        rows = int(rows / factor)
        cols = int(cols / factor)

        new_size = (cols, rows)
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image


def get_generation_config(repetition_penalty):
    """Create generation config with specified repetition penalty"""
    return dict(
        max_new_tokens=1024,
        do_sample=False,
        num_beams=4,
        repetition_penalty=repetition_penalty,
    )


# def get_dcm_from_bucket(gcp_bucket_path, date="22JUL2024"):
#     base = f"gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/{date}/"
#     gcp_bucket_path = base + gcp_bucket_path

#     path_parts = gcp_bucket_path.split("/")
#     bucket_name = path_parts[2]
#     blob_path = "/".join(path_parts[3:])

#     storage_client = storage.Client()

#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(blob_path)

#     dicom_data = blob.download_as_bytes()

#     dicom_file = pydicom.dcmread(BytesIO(dicom_data))

#     return dicom_file


def get_dcm_from_local(local_path):
    try:
        dicom_file = pydicom.dcmread(local_path)
    except:
        dicom_file = pydicom.dcmread(local_path, force=True)
        dicom_file = dicom_compression_utils.handle_dicom_compression(dicom_file)

    return dicom_file


def load_image(image_file, input_size=448, max_num=12, root_path=""):
    if "dcm" in image_file:
        # dcm_data = get_dcm_from_bucket(image_file)
        dcm_data = get_dcm_from_local(image_file)
        image = dcm_2_rgb(dcm_data, image_file)
    else:
        # Check if R2 client is available and R2 streaming is enabled
        r2_client = get_r2_client()
        if r2_client is not None and R2_STREAM_FROM_R2:
            # Remove root path prefix if present
            image_path = image_file.replace(root_path, '', 1) if image_file.startswith(root_path) else image_file
            image = _fetch_image_from_r2(image_path)
        else:
            image = Image.open(image_file).convert("RGB")

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


@torch.inference_mode()
def generate_output(lines, model, tokenizer, output_path, rank, generation_config):
    results = []
    times = []

    for line in tqdm(lines, desc=f"Processing_{rank}"):
        # Parse the line as a JSON object
        start_time = time.time()
        entry = json.loads(line)

        # Process the JSON object (e.g., print it)
        image_paths = entry["image"]
        image_paths = [
            each.replace("/mnt/all_data", "/mnt/all-data") for each in image_paths
        ]
        entry["image"] = image_paths
        try:

            pixel_values_list = [
                load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                for image_path in image_paths
            ]
            pixel_values = torch.cat(pixel_values_list, dim=0)
            num_patches_list = [
                pixel_values.size(0) for pixel_values in pixel_values_list
            ]

            query, truth_report = entry["conversations"]
            query = query["value"]
            truth_report = truth_report["value"]

            response = model.module.chat(
                tokenizer,
                pixel_values,
                query,
                generation_config,
                num_patches_list=num_patches_list,
            )

            # Clear GPU cache after each inference
            del pixel_values
            del pixel_values_list
            torch.cuda.empty_cache()

        except Exception as e:
            print(f">>>>>>>>>Error: {e}")
            # Clear cache even on error
            torch.cuda.empty_cache()
            continue

        #entry["prompt"] = query
        #entry["rad_report"] = truth_report
        entry["new_generated_report_local"] = response

        print(">>>>>> generated report")
        print(response)

        results.append(entry)

        end_time = time.time()
        times.append(end_time - start_time)

    # Save results for this rank
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    jsonl_output_path = output_path.replace('.pkl', '.jsonl')
    with open(jsonl_output_path, 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')


def aggregate_results(world_size, description, output_dir):
    aggregated_results = []

    for rank in range(world_size):
        output_path = f"{output_dir}/{description}_{rank}.pkl"
        with open(output_path, "rb") as f:
            aggregated_results.extend(pickle.load(f))

    # Save the final aggregated results
    final_output_path = f"{output_dir}/{description}-final_output.pkl"
    with open(final_output_path, "wb") as f:
        pickle.dump(aggregated_results, f)

    print(f"Aggregated results saved to {final_output_path}")

def run_inference_for_penalty(repetition_penalty, base_description):
    """Run inference for a specific repetition penalty"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create description with penalty value
    description = f"{base_description}_p{repetition_penalty}"

    if rank == 0:
        print(
            f"Running inference with repetition_penalty={repetition_penalty}, description={description}"
        )

    # Get generation config for this penalty
    generation_config = get_generation_config(repetition_penalty)

    if rank == 0:
        print(f"generation_config: {generation_config}")

    # Set up paths
    test_jsonl = "/home/ruian/projects/all_data_cleaning/matt_csv_polish/data/0917_prod.jsonl"
    test_jsonl = "/home/ruian/projects/all_data_cleaning/matt_csv_polish/data/0917_prod_no_label.jsonl"
    test_jsonl = "/home/ruian/projects/all_data_cleaning/prod_csv_polish/prod_data_v2_with_label_mpo.jsonl"
    test_jsonl = "/root/projects/InternVL-3x/internvl_chat/data/1118_hand_degen_png_selected_clean_words.jsonl"
    test_jsonl = "/root/projects/InternVL-3x/internvl_chat/data/hand_data.jsonl"
    test_jsonl = "/root/projects/InternVL-3x/internvl_chat/data/length_eval_subset.jsonl"

    # checkpoint_dir = "/mnt/pngs/internvl_weights/internvl3_chimera_20250906_075059_1e-5_consolidated_labels-0904"
    # checkpoint_dir = "/mnt/pngs/internvl_weights/internvl3_chimera_20250913_021402_1e-5_labels_spine_only-0912-8B"
    # checkpoint_dir = "/home/ruian/vlm_ckpts_v2/labels/internvl3_chimera_20250913_021402_1e-5_labels_spine_only-0912-8B"

    #checkpoint_dir = "/home/ruian/vlm_ckpt_v2.0/label/internvl3_chimera_20251009_004033_1e-5_consolidated_labels-1009-38B-8B/"
    #checkpoint_dir = "/home/ruian/vlm_ckpt_v2.0/no-label/internvl3_chimera_20251011_031636_1e-5_no_labels-1009-38B-8B"

    checkpoint_dir = "/root/projects/InternVL-3x/internvl_chat/training/internvl3_8b_20251124_231837_1e-5_sf_1124_hand_length_15"
    checkpoint_dir = "/root/projects/InternVL-3x/internvl_chat/training/internvl3_8b_20251124_234949_1e-5_sf_1124_hand_length_20"
    checkpoint_dir = "/root/projects/InternVL-3x/internvl_chat/training/internvl3_8b_20251124_235336_1e-5_sf_1124_hand_length_22"

    output_dir = (
        f"/root/projects/InternVL-3x/internvl_chat/test_data/pkls/{description}"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if rank == 0:
            print(f"Directory '{output_dir}' created.")

    checkpoints = sorted(
        [
            os.path.join(checkpoint_dir, ckpt)
            for ckpt in os.listdir(checkpoint_dir)
            if ckpt.startswith("checkpoint-")
        ],
        key=lambda x: int(x.split("-")[-1]),
    )

    if rank == 0:
        print(f"Found {len(checkpoints)} checkpoints to evaluate.")

    for checkpoint in checkpoints:
        # if not "23688" in checkpoint:
        #     if rank == 0:
        #         print(f"Skipping {checkpoint}")
        #     continue

        suffix = checkpoint.split("/")[-1]
        if rank == 0:
            print(f"Loading model from {checkpoint}, with a suffix of {suffix}")

        output_path = f"{output_dir}/{suffix}/{rank}.pkl"

        if os.path.exists(output_path):
            if rank == 0:
                print(f"Warning: {output_path} already exists. Skipping...")
            continue

        model = InternVLChatModel.from_pretrained(
            checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(f"cuda:{rank}")

        model.eval()
        model = DDP(model, device_ids=[rank], output_device=rank)

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, use_fast=False
        )

        # Partition dataset among GPUs
        with open(test_jsonl, "r") as file:
            all_lines = file.readlines()

        local_lines = all_lines[rank::world_size]

        os.makedirs(f"{output_dir}/{suffix}", exist_ok=True)

        if rank == 0:
            print(f"saving world-{rank} to {output_path}")

        generate_output(
            local_lines, model, tokenizer, output_path, rank, generation_config
        )

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()


def main():
    # Initialize distributed processing
    print("Initializing distributed processing...")
    init_distributed()
    print("Distributed initialization complete.")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Running inference with {world_size} GPUs...")

        # Log R2 configuration status
        if R2_ENDPOINT_URL and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
            print(f"R2 S3 support enabled: bucket={R2_BUCKET_NAME}, prefix={R2_BUCKET_PNG_PREFIX}")
        else:
            print("R2 S3 support disabled: using local file system")

    if len(sys.argv) < 2:
        print("Usage: python3 -m intern_evaluation.py <base_description>")
        sys.exit(1)

    base_description = sys.argv[1]

    # Define the range of repetition penalties to test
    # penalty_values = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    penalty_values = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

    penalty_values = [1.5]

    if rank == 0:
        print(f"Will run inference for repetition penalties: {penalty_values}")

    # Run inference for each penalty value
    for penalty in penalty_values:
        if rank == 0:
            print(f"\n{'=' * 50}")
            print(f"Starting inference for repetition_penalty = {penalty}")
            print(f"{'=' * 50}")

        # Synchronize all processes before starting each penalty run
        # dist.barrier()

        try:
            run_inference_for_penalty(penalty, base_description)
            if rank == 0:
                print(f"Completed inference for repetition_penalty = {penalty}")
        except Exception as e:
            if rank == 0:
                print(f"Error during inference for repetition_penalty = {penalty}: {e}")
            continue

        # Synchronize all processes after completing each penalty run
        # dist.barrier()

    if rank == 0:
        print("All repetition penalty experiments completed!")


if __name__ == "__main__":
    main()
