import json
import os
import pickle
import shutil
import argparse
from collections import defaultdict
from tqdm import tqdm

all_labels = [
    'Support Devices',
    'No Findings',
    'Cardiomegaly',
    'Atelectasis',
    'Airspace Opacity',
    'Pleural Effusion',
    'Edema',
    'Pneumonia',
    'Fracture',
    'Lung Lesion',
    'Consolidation',
    'Pleural Other',
    'Pneumothorax',
    'Enlarged Cardiomediastinum'
]


def aggregate_results_one_folder(world_size, description, output_dir):
    aggregated_results = []

    for rank in range(world_size):
        output_path = os.path.join(output_dir, f"{rank}.pkl")
        with open(output_path, "rb") as f:
            aggregated_results.extend(pickle.load(f))

    final_output_path = os.path.join(output_dir, f"{description}-final_output.pkl")
    with open(final_output_path, "wb") as f:
        pickle.dump(aggregated_results, f)


    # Save as JSONL
    final_jsonl_path = os.path.join(output_dir, f"{description}-final_output.jsonl")

    with open(final_jsonl_path, 'w') as f:
        for entry in aggregated_results:
            f.write(json.dumps(entry) + '\n')

    print(f"Aggregated results saved to {final_output_path} and {final_jsonl_path}")


def aggregate_results(world_size, description, output_dir, base_dir):
    aggregated_results = []
    aggregate_results_per_label = defaultdict(list)

    for rank in tqdm(range(world_size), desc=f"Processing {description}"):
        output_path = os.path.join(output_dir, f"{rank}.pkl")

        if not os.path.exists(output_path):
            print(f"Warning: {output_path} does not exist. Skipping...")
            continue

        with open(output_path, "rb") as f:
            results = pickle.load(f)
            aggregated_results.extend(results)

            for entry in results:
                labels = entry.get("labels", [])
                for label in labels:
                    if label in all_labels:
                        aggregate_results_per_label[label].append(entry)

    # Save aggregated results
    final_output_path = os.path.join(base_dir, f"{description}.pkl")
    final_output_label_path = os.path.join(base_dir, f"{description}_label.pkl")

    print(f"Total entries: {len(aggregated_results)}")
    with open(final_output_path, "wb") as f:
        pickle.dump(aggregated_results, f)

    print(f"Labels aggregated: {len(aggregate_results_per_label)}")
    with open(final_output_label_path, "wb") as f:
        pickle.dump(aggregate_results_per_label, f)


    # Save aggregated results as JSONL

    final_jsonl_path = os.path.join(base_dir, f"{description}.jsonl")

    with open(final_jsonl_path, 'w') as f:
        for entry in aggregated_results:
            f.write(json.dumps(entry) + '\n')

    # Save per-label results as JSONL

    final_jsonl_label_path = os.path.join(base_dir, f"{description}_label.jsonl")

    with open(final_jsonl_label_path, 'w') as f:
        for label, entries in aggregate_results_per_label.items():
            for entry in entries:
                # Add label info to make it easier to filter later
                entry_with_label = entry.copy()
                entry_with_label['_aggregated_label'] = label
                f.write(json.dumps(entry_with_label) + '\n')

    print(f"Aggregated results saved to {final_output_path} and {final_jsonl_path}")
    print(f"Per-label results saved to {final_output_label_path} and {final_jsonl_label_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate results from distributed pickle outputs.")
    parser.add_argument("--path", type=str, required=True, help="Base directory containing result subfolders.")
    parser.add_argument("--world_size", type=int, default=8, help="Number of distributed ranks.")
    parser.add_argument("--remove", action="store_true", help="Remove subdirectories after aggregation.")
    args = parser.parse_args()

    base_dir = args.path
    world_size = args.world_size

    for subdir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, subdir)
        if os.path.isdir(full_path):
            description = subdir
            print(f"Processing directory: {description}")
            aggregate_results(world_size, description, full_path, base_dir)

            if args.remove:
                shutil.rmtree(full_path)
                print(f"Removed directory: {full_path}")
