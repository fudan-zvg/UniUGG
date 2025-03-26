import os
import json
from tqdm import tqdm
from pathlib import Path


def main():
    mix_ratios = {
        "rxr_11k": 1.0,
        "scannet_7799k": 1.0,
        "scannetpp_5941k": 1.0,
        "structured3d_2523k": 1.0,
        # "2d_data": xx,
    }
    # mix_ratios = {
    #     "rxr_11k": 0.01,
    #     "scannet_7799k": 0.001,
    #     "scannetpp_5941k": 0.001,
    #     "structured3d_2523k": 0.001,
    #     # "2d_data": xx,
    # }

    qa_type_ratios = {
        "sentence": 1.0,
        "select": 0.1,
        "fill": 0.1,
        "judge": 0.1,
    }

    base_dir = Path("data_jsons")
    final_data_json = {}
    total_len = 0

    for dataset_name in tqdm(mix_ratios.keys(), desc="Loading datasets"):
        json_path = base_dir / f"{dataset_name}.json"
        if not json_path.exists():
            print(f"[Warning] File not found: {json_path}")
            continue

        with json_path.open("r", encoding="utf-8") as f:
            data_json = json.load(f)

        for task_key, item in data_json.items():
            task_type = task_key.split("_")[-1]

            if task_type not in qa_type_ratios:
                print(f"[Warning] Unknown task type: {task_type}, skipping.")
                continue

            
            type_ratio = qa_type_ratios[task_type]
            repeat_time = mix_ratios[dataset_name] * type_ratio

            item["repeat_time"] = repeat_time
            item["length_with_repeat"] = int(item["length"] * repeat_time)

            final_data_json[task_key] = item
            total_len += item["length_with_repeat"]

    mix_str = "_".join([
        f"{str(ratio).rstrip('0').rstrip('.') if ratio != 1.0 else '1'}{k.split('_')[0]}"
        for k, ratio in mix_ratios.items()
    ])
    output_file = base_dir / f"{mix_str}_{total_len // 1_000_000}m.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(final_data_json, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Finished! Total length: {total_len}")
    print(f"📄 Output saved to: {output_file}")


if __name__ == "__main__":
    main()
