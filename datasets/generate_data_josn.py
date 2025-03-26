import os
import json
from tqdm import tqdm
from pathlib import Path

def process_dataset(dataset: str, base_dir: str = "spar"):
    train_qa_dir = Path(base_dir) / dataset / "qa_jsonl" / "train"
    dataset_json_mapping = {}

    task_folders = [f for f in train_qa_dir.iterdir() if f.is_dir()]
    for task_folder in tqdm(task_folders, desc=f"Processing {dataset}", unit="task"):
        for sub_task_folder in task_folder.iterdir():
            if not sub_task_folder.is_dir():
                continue
            for qa_file in sub_task_folder.iterdir():
                if not qa_file.is_file() or qa_file.suffix != ".jsonl":
                    continue 

                try:
                    with qa_file.open("r", encoding="utf-8") as file:
                        qa_pairs = [json.loads(line) for line in file]
                except Exception as e:
                    print(f"Error reading {qa_file}: {e}")
                    continue

                try:
                    expected_length = int(qa_file.stem.split("_")[-1])
                    actual_length = len(qa_pairs)
                except ValueError:
                    print(f"Skipping {qa_file}, invalid filename format.")
                    continue

                if expected_length != actual_length:
                    new_name = qa_file.name.replace(str(expected_length), str(actual_length))
                    new_path = sub_task_folder / new_name
                    try:
                        qa_file.rename(new_path)
                        print(f"Renamed {qa_file.name} -> {new_name}")
                    except Exception as e:
                        print(f"Failed to rename {qa_file}: {e}")
                        continue

                key_name = f"{dataset}_{task_folder.name}_{sub_task_folder.name}"
                dataset_json_mapping[key_name] = {
                    "root": str(Path(base_dir) / dataset / "images"),
                    "annotation": str(qa_file),
                    "repeat_time": 1,
                    "length": actual_length,
                }

    total_length_k = sum(entry["length"] for entry in dataset_json_mapping.values()) // 1000

    output_dir = Path("data_jsons")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{dataset}_{total_length_k}k.json"
    
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(dataset_json_mapping, file, indent=4)
    
    print(f"Finished processing {dataset}, saved to {output_file}")

if __name__ == "__main__":
    dataset_list = [
        "rxr",
        "scannet",
        "scannetpp",
        "structured3d",
    ]
    for dataset in dataset_list:
        process_dataset(dataset)

