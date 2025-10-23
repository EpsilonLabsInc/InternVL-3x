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
import torch.distributed as dist
import torchvision.transforms as T
# from google.cloud import storage
from internvl.model.internvl_chat import InternVLChatModel
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoTokenizer
import math

def init_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


sys.path.append("/home/eric/projects/InternVL-3x/internvl_chat")
sys.path.append("/home/eric/projects/epsutils")

from epsutils.dicom import dicom_utils, dicom_compression_utils
from epsutils.image import image_utils

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
    image = dicom_utils.get_dicom_image_from_dataset(dcm_data, custom_windowing_parameters={"window_center": 0, "window_width": 0})
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


# generation_config = dict(
#     max_new_tokens=1024,
#     do_sample=True,
#     temperature=0.5,
#     top_k=100,
#     num_beams=2,
#     repetition_penalty=1.5,
# )

# generation_config = dict(
#     max_new_tokens=1024,
#     do_sample=True,
#     temperature=0.5,
#     top_k=100,
#     num_beams=2,
#     repetition_penalty=1.5
# )

generation_config = dict(
            max_new_tokens=1024,
            do_sample=False,
            num_beams=4,
            repetition_penalty=2.8,
        )

print("generation_config: ", generation_config)

def get_dcm_from_bucket(gcp_bucket_path, date="22JUL2024"):
    base = f"gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/{date}/"
    gcp_bucket_path = base + gcp_bucket_path

    path_parts = gcp_bucket_path.split("/")
    bucket_name = path_parts[2]
    blob_path = "/".join(path_parts[3:])

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    dicom_data = blob.download_as_bytes()

    dicom_file = pydicom.dcmread(BytesIO(dicom_data))

    return dicom_file

def get_dcm_from_local(local_path):

    try:
        dicom_file = pydicom.dcmread(local_path)
    except:
        dicom_file = pydicom.dcmread(local_path, force=True)
        dicom_file = dicom_compression_utils.handle_dicom_compression(dicom_file)

    return dicom_file

def load_image(image_file, input_size=448, max_num=12):
    if "dcm" in image_file:
        # dcm_data = get_dcm_from_bucket(image_file)
        dcm_data = get_dcm_from_local(image_file)
        image = dcm_2_rgb(dcm_data, image_file)
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
def generate_output(lines, model, tokenizer, output_path, rank):
    results = []
    times = []

    for line in tqdm(lines, desc=f"Processing_{rank}"):
        # print('----------------------------')

        # Parse the line as a JSON object
        start_time = time.time()
        entry = json.loads(line)

        # Process the JSON object (e.g., print it)
        image_paths = entry["image"]
        image_paths = [
            each.replace(
                "/mnt/all_data", "/mnt/all-data"
            )
            for each in image_paths
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

            # response = model.module.chat(
            #     tokenizer,
            #     pixel_values,
            #     query,
            #     generation_config,
            #     num_patches_list=num_patches_list,
            # )
            # print(f"at rank {rank}, Good!!!!!!!!")
        except Exception as e:
            print(f">>>>>>>>>Error: {e}")
            # print(entry)
            # print("Error query:", query)
            # print("Error truth_report:", truth_report)
            continue

        # result = {"idx": entry["idx"], "truth": truth_report, "generated": response}
        # results.append(result)

        # print(">>>")
        # print(truth_report)
        # print("<<<")
        # print(response)

        entry['prompt'] = query
        entry["rad_report"] = truth_report
        entry["generated_report"] = response

        print(">>>>>> generated report")
        print(response)

        # print(report_formation(response))

        results.append(entry)

        end_time = time.time()
        times.append(end_time - start_time)

    # Save results for this rank
    with open(output_path, "wb") as f:
        pickle.dump(results, f)


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

from openai import AzureOpenAI # need this added to requirements.txt, openai==1.68.2

def report_formation(report):
    c_spine_text_1 = 'FINDINGS:\n\nVertebrae: No acute fracture. Large posterior osteophytes at L5-S1 likely results in neural foraminal stenosis. No\nposterior element or facet joint abnormality is detected.\n\nSacrum/coccyx: Normal as visualized. No acute fracture.\n\nDisc spaces: Severe intervertebral disc space narrowing with endplate sclerosis at L5-S1 is seen. Large posterior\nosteophyte is detected.\n\nSoft tissues: Normal.\n\nIMPRESSION:\n\nSevere degenerative disc changes at L5-S1 with large posterior osteophyte. MRI may be considered to assess for\nneural foraminal and central canal stenosis.'
    c_spine_text_2 = 'FINDINGS:\nALIGNMENT: Grade 1 retrolisthesis of L1 on L2 and L2 on L3. Miner dextrocurvature of the lumbar spine\n\nSPINE: Vertebral body heights are preserved. . Moderate C5-C6 spondylosis and mild C6-C7 spondylosis.\nMultilevel cervical spine facet arthropathy.\n\nPostsurgical change of L4-L5 and L5-S1. Moderate to severe L1-L2 spondylosis. Multilevel lumbar spine facet\narthropathy.\n\nVISUALIZED LUNGS: No significant abnormalities.\nVISUALIZED ABDOMEN: No significant abnormalities.\n\nOTHER: None.\n\nIMPRESSION:\n\nMultilevel spondylosis and postsurgical change of the lumbar spine.\n'
    l_spine_text = 'FINDINGS:\n\nALIGNMENT: Minor grade 1 retrolisthesis of L2 on L3 and mild grade 1 anterolisthesis of L4 on LS.\n\nSPINE: Vertebral body heights are preserved. Multilevel spondylosis and multilevel facet arthropathy. Multilevel\nthoracic spine spondylosis. No acute fracture or pars defect.\n\nSACRUM AND SI JOINTS: No significant abnormalities.\nVISUALIZED ABDOMEN: No significant abnormalities.\nIMPRESSION:\n\nMultilevel spondylosis.'
    ankle_text = 'FINDINGS:\n\nBones/joints: A 1.3 cm osteochondral lesion in the central talar dome is again seen. Nonhealed avulsion fracture\nof the distal fibula is identified. 2 surgical screws fixate the talus. Minimal pes cavus is identified on the lateral\nprojection of the left foot. There is narrowing with small osteophytes at the first metatarsophalangeal joint space.\nNo acute fracture. No dislocation.\n\nSoft tissues: Normal. No radiopaque foreign body.\n\nIMPRESSION:\n\n1. No acute findings.\n\n2. Minimal pes cavus is detected.\n\n3. Osteochondral lesion at the central talar dome. Postsurgical changes of the talus.\n\n4. Nonhealed avulsion fracture of the distal fibula.\n\n5. Arthritic changes at the first metatarsal phalangeal joint space.'
    foot_text = 'FINDINGS:\n\nBones/joints: Narrowing of the first metatarsophalangeal joint space is detected. Superior and inferior calcaneal\nspurs are noted. No acute fracture. No dislocation.\n\nSoft tissues: Normal. No radiopaque foreign body.\nIMPRESSION:\n1. Superior and inferior calcaneal spurs.\n\n2. Narrowing of the first metatarsophalangeal joint space.\n'
    shoulder_text = 'FINDINGS/IMPRESSION:\n\nBONE MINERALIZATION: Normal.\n\nBONES: No acute fracture or dislocation. The glenohumeral and acromioclavicular joints are intact. No significant\ndegenerative change. No erosive or aggressive-appearing osseous lesion.\n\nSOFT TISSUES: Soft tissues are unremarkable. The visualized portions of the lung are clear.'
    chest_text = 'FINDINGS:\n\nLUNGS: No pleural effusion, focal consolidation, or pneumothorax.\nMEDIASTINUM AND HILA: Mediastinal and hilar contours are normal.\nHEART: Cardiac silhouette is normal in size.\n\nPULMONARY VASCULARITY: Normal.\n\nBONES: No acute osseous abnormalities. Multilevel spondylosis.\nOTHER: None.\n\nIMPRESSION:\n\nNo acute radiographic abnormality of the chest.'

    text_examples = {
        "chest": chest_text,
        "shoulder": shoulder_text,
        "foot": foot_text,
        "ankle": ankle_text,
        "l_spine": l_spine_text,
        "c_spine_1": c_spine_text_1,
        "c_spine_2": c_spine_text_2,
    }

    endpoint = "https://epsilonlabs.openai.azure.com/"
    deployment = "gpt-4.1"

    subscription_key = ""

    api_version = "2024-12-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful radiologist.",
            },
            {
                "role": "user",
                "content": f"Please convert the input xray medical report to styles inclued here {text_examples}. Only change the style to have a key word followed by description. Do not alter meaning of the original report. Only output the converted report. Input xray medical report is {report}",
            }
        ],
        max_completion_tokens=800,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment
    )

    return response.choices[0].message.content



def main():
    # Initialize distributed processing
    print("Initializing distributed processing...")
    init_distributed()
    print("Distributed initialization complete.")


    rank = dist.get_rank()  # Get the rank of the current process
    world_size = dist.get_world_size()  # Total number of processes

    if rank == 0:  # Only rank 0 prints logs
        print(f"Running inference with {world_size} GPUs...")


    # test_jsonl = "/mnt/data/eric/cr_all3/combined_output_test_1129.jsonl" # with labels
    # test_jsonl = "/mnt/data/eric/cr_all3/combined_output_test_no_label_1122.jsonl" # no labels
    # test_jsonl = "/home/eric/projects/InternVL-Epsi/internvl_chat/test_data/11192024_test_selected_136.jsonl"
    # test_jsonl = "/home/eric/projects/InternVL-Epsi/internvl_chat/test_data/11192024_test_selected_136_nolabel_nebius.jsonl"
    # test_jsonl = "/home/eric/projects/InternVL-Epsi/internvl_chat/test_data/11192024_test_selected_136_system_msg.jsonl"
    # test_jsonl = "/home/eric/projects/InternVL-Epsi/internvl_chat/test_data/11192024_test_selected_136_system_msg_random_synonym.jsonl"
    # test_jsonl = "/home/eric/projects/InternVL-Epsi/internvl_chat/test_data/combined_output_test_no_label_01222025_nebius.jsonl"
    # test_jsonl = "/home/eric/projects/InternVL-Epsi/internvl_chat/test_data/combined_output_test_no_label_01222025_nebius_filtered.jsonl"
    test_jsonl = "/home/eric/projects/InternVL-Epsi/output/jsonl/mimic2/03202025_atmost2images_no_label_test.jsonl"
    test_jsonl = "/home/eric/projects/InternVL-Epsi/output/jsonl/other_parts/0403_test.jsonl"
    test_jsonl = "/home/eric/projects/InternVL-Epsi/output/jsonl/gradient/0410_gradient_all_labels_test.jsonl"
    test_jsonl = "/home/eric/projects/InternVL-Epsi/output/jsonl/gradient/0416_gradient_all_no_label_test.jsonl"
    test_jsonl = "/home/eric/projects/InternVL-Epsi/output/jsonl/mimic2/04212025_atmost4images_no_label_test_interview.jsonl"
    test_jsonl = "/home/eric/projects/InternVL-3x/output/jsonl/gradient/all_test_0428_updated.jsonl"
    test_jsonl = "/home/eric/projects/InternVL-3x/output/jsonl/gradient/all_chest_0507_test.jsonl"
    test_jsonl = "/home/eric/projects/InternVL-3x/output/jsonl/all/all_06082025_no_labels_test.jsonl"
    test_jsonl = "/home/eric/projects/all_data_cleaning/sampled_1000_nolabels.jsonl"
    test_jsonl = "/home/eric/projects/all_data_cleaning/sampled_1000_labels.jsonl"
    # test_jsonl = "/home/eric/projects/all_data_cleaning/sampled_simonmed.jsonl"
    # test_jsonl = "/home/eric/projects/all_data_cleaning/sampled_simonmed_added_labels.jsonl"
    # test_jsonl = "/home/eric/projects/all_data_cleaning/sampled_simonmed_added_labels_no_findings.jsonl"
    # test_jsonl = "/home/eric/projects/all_data_cleaning/sampled_simonmed_removed_labels_fracture.jsonl"
    # test_jsonl = "/home/eric/projects/all_data_cleaning/sampled_simonmed_fracture.jsonl"
    test_jsonl = "/home/eric/projects/all_data_cleaning/0818_polished_labels.jsonl"

    # test_jsonl = "/home/eric/projects/InternVL-3x/output/jsonl/all/all_08162025_labels_test_gpt_body_part_sample_1000.jsonl"
    # test_jsonl = "/home/eric/projects/all_data_cleaning/0818_polished_nolabels.jsonl"
    # test_jsonl = "/home/eric/projects/all_data_cleaning/sampled_1000_chest_labels.jsonl"

    print(f"Using test JSONL file: {test_jsonl}")

    # test_jsonl = "/home/eric/projects/InternVL-Epsi/internvl_chat/test_data/combined_output_test_1129_add_random_label_nebius.jsonl"
    # test_jsonl = "/home/eric/projects/InternVL-Epsi/internvl_chat/test_data/combined_output_test_1129_add_random_label_nebius_nolabel.jsonl"
    # test_jsonl = "/home/eric/projects/InternVL-Epsi/internvl_chat/test_data/combined_output_test_1129_gradient_only_nebius.jsonl"

    # checkpoint_dir = "/mnt/training/internvl3_chimera_20250609_233409_1e-5_epsilon_all_0608" # no_laqbel
    # checkpoint_dir = "/home/eric/projects/InternVL-3x/internvl_chat/training/chimera_26b-2b_20250411_072139_1e-5_2.5_gradien_allchest_2images"
    # checkpoint_dir = "/mnt/training/internvl3_chimera_20250617_045312_1e-5_epsilon_all_0608" # 14 label
    # checkpoint_dir = "/mnt/training/internvl3_chimera_20250626_035443_1e-5_epsilon_all_0608/" # new label
    # checkpoint_dir = "/mnt/training/internvl_weights/useful/internvl3_chimera_20250630_190606_1e-5_epsilon_all_0608" # new label
    # checkpoint_dir = "/mnt/training/internvl3_chimera_20250707_220248_1e-5_epsilon_all_0706"

    # checkpoint_dir = "/mnt/training/internvl3_chimera_20250711_164248_1e-5_epsilon_chest_0711" # chest-data-chest-labels
    # checkpoint_dir = "/mnt/training/internvl3_chimera_20250713_173014_1e-5_epsilon_nonchest_0714" # non-chest-data-non-chest-labels
    # checkpoint_dir = "/mnt/training/internvl_weights/simonmed/internvl3_chimera_20250721_182717_1e-5_simonmed_0721"

    # checkpoint_dir = "/mnt/pngs/internvl_weights/internvl3_chimera_20250810_083849_1e-5_0810_no_label_gpt_bodypart"
    checkpoint_dir = "/mnt/pngs/internvl_weights/internvl3_chimera_20250817_021521_1e-5_0816_label_gpt_bodypart"
    checkpoint_dir = "/mnt/pngs/internvl_weights/internvl3_chimera_20250821_014752_1e-5_0820_label_gpt_bodypart_only_findings"
    checkpoint_dir = "/mnt/pngs/internvl_weights/internvl3_chimera_20250828_212209_1e-5_0828_label_gpt_bodypart"
    checkpoint_dir = "/mnt/pngs/internvl_weights/internvl3_chimera_20250906_075059_1e-5_consolidated_labels-0904"

    if len(sys.argv) < 2:
        print("Usage: python3 -m intern_evaluation.py <description>")
        sys.exit(1)

    description = sys.argv[1]
    # output_dir = f"/mnt/data/eric/internvl2/pkls/{description}"
    output_dir = f"/home/eric/projects/InternVL-3x/internvl_chat/test_data/pkls/{description}"

    if not os.path.exists(output_dir):
        # Proceed with creating the directory if needed or continue processing
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
        print(f"Directory '{output_dir}' created.")

    checkpoints = sorted(
        [
            os.path.join(checkpoint_dir, ckpt)
            for ckpt in os.listdir(checkpoint_dir)
            if ckpt.startswith("checkpoint-")
        ],
        key=lambda x: int(x.split("-")[-1])
    )

    print(f"Found {len(checkpoints)} checkpoints to evaluate. They are:")
    print(checkpoints)

    for checkpoint in checkpoints:

        if not "18777" in checkpoint:
            print(f"Skipping {checkpoint}")
            continue

        suffix = checkpoint.split("/")[-1]
        print(f"Loading model from {checkpoint}, with a suffix of {suffix} at rank {rank}")

        output_path = f"{output_dir}/{suffix}/{rank}.pkl"

        if os.path.exists(output_path):
            print(f"Warning: {output_path} already exists. Skipping...")
            continue

        model = InternVLChatModel.from_pretrained(
            checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(f"cuda:{rank}")

        # base_model = InternVLChatModel.from_pretrained(
        #     "./pretrained/InternVL2_5-26B-MPO",  # Path to the original base model (non-LoRA)
        #     torch_dtype=torch.bfloat16,
        #     device_map=None)

        # from peft import PeftModel

        # print("loading lora parts")
        # model = PeftModel.from_pretrained(base_model,
        #                                   checkpoint,
        #                                   is_local_files_only=True)

        model.eval()

        # print(f"at rank {rank}, model loaded from {checkpoint}")
        # print(f">>>><mode is {model}")
        model = DDP(model, device_ids=[rank], output_device=rank)
        # print(f"at rank {rank}, model wrapped in DDP")
        # print(f"<<<<<mode is {model}")

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, use_fast=False
        )

        # Partition dataset among GPUs
        with open(test_jsonl, "r") as file:
            all_lines = file.readlines()

        # Each GPU gets its portion of data
        local_lines = all_lines[rank::world_size]
        # print(f"at {rank}, local_lines: {local_lines[:10]}")

        os.makedirs(f"{output_dir}/{suffix}", exist_ok=True)

        print(f"saving world-{rank} to {output_path}")

        generate_output(local_lines, model, tokenizer, output_path, rank)

        # if rank == 0:  # Aggregate results on rank 0
        #     aggregate_results(world_size, description, output_dir=output_dir)

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
