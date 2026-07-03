import os
import ijson
import json
from pathlib import Path

def filter_dataset(dataroot: str, original_meta_dir: str, new_meta_dir: str):
    dataroot = Path(dataroot)
    orig_dir = Path(original_meta_dir)
    new_dir = Path(new_meta_dir)
    new_dir.mkdir(parents=True, exist_ok=True)

    print("1. Scanning for valid CAM_FRONT files...")
    cam_front_dir = dataroot / 'samples' / 'CAM_FRONT'
    valid_filenames = set()
    if cam_front_dir.exists():
        for f in cam_front_dir.iterdir():
            if f.is_file():
                # Store relative path exactly as nuScenes expects
                valid_filenames.add(f"samples/CAM_FRONT/{f.name}")
    print(f"Found {len(valid_filenames)} valid CAM_FRONT files locally.")

    print("2. Finding valid sample tokens...")
    valid_sample_tokens = set()
    with open(orig_dir / 'sample_data.json', 'rb') as f:
        for item in ijson.items(f, 'item', use_float=True):
            if item['filename'] in valid_filenames:
                valid_sample_tokens.add(item['sample_token'])
    print(f"Found {len(valid_sample_tokens)} valid samples.")

    print("3. Collecting related tokens...")
    valid_ego_pose_tokens = set()
    valid_calib_tokens = set()
    
    # We must collect ALL sample_data belonging to valid samples (e.g. radars, other cams)
    valid_sample_data_tokens = set()
    
    with open(orig_dir / 'sample_data.json', 'rb') as f:
        filtered_sample_data = []
        for item in ijson.items(f, 'item', use_float=True):
            if item['sample_token'] in valid_sample_tokens:
                filtered_sample_data.append(item)
                valid_sample_data_tokens.add(item['token'])
                valid_ego_pose_tokens.add(item['ego_pose_token'])
                valid_calib_tokens.add(item['calibrated_sensor_token'])
                
    with open(new_dir / 'sample_data.json', 'w') as f:
        json.dump(filtered_sample_data, f)
    print(f"Saved {len(filtered_sample_data)} sample_data records.")

    print("4. Filtering sample.json...")
    valid_scene_tokens = set()
    with open(orig_dir / 'sample.json', 'rb') as f:
        filtered_samples = []
        for item in ijson.items(f, 'item', use_float=True):
            if item['token'] in valid_sample_tokens:
                filtered_samples.append(item)
                valid_scene_tokens.add(item['scene_token'])
    with open(new_dir / 'sample.json', 'w') as f:
        json.dump(filtered_samples, f)

    print("5. Filtering sample_annotation.json (This is the huge one)...")
    valid_instance_tokens = set()
    with open(orig_dir / 'sample_annotation.json', 'rb') as f:
        filtered_annotations = []
        for item in ijson.items(f, 'item', use_float=True):
            if item['sample_token'] in valid_sample_tokens:
                filtered_annotations.append(item)
                valid_instance_tokens.add(item['instance_token'])
    with open(new_dir / 'sample_annotation.json', 'w') as f:
        json.dump(filtered_annotations, f)
    print(f"Saved {len(filtered_annotations)} annotations.")

    print("6. Filtering remaining tables...")
    def filter_table(name, valid_set, extract_key=None, target_set=None):
        with open(orig_dir / f"{name}.json", 'rb') as f:
            filtered = []
            for item in ijson.items(f, 'item', use_float=True):
                if item['token'] in valid_set:
                    filtered.append(item)
                    if extract_key and target_set is not None:
                        target_set.add(item[extract_key])
        with open(new_dir / f"{name}.json", 'w') as f:
            json.dump(filtered, f)
        print(f"Saved {len(filtered)} records to {name}.json")

    filter_table('instance', valid_instance_tokens)
    filter_table('ego_pose', valid_ego_pose_tokens)
    filter_table('calibrated_sensor', valid_calib_tokens)
    
    valid_log_tokens = set()
    filter_table('scene', valid_scene_tokens, 'log_token', valid_log_tokens)
    
    filter_table('log', valid_log_tokens)

    print("7. Copying small tables...")
    for name in ['category', 'attribute', 'visibility', 'sensor', 'map']:
        with open(orig_dir / f"{name}.json", 'rb') as f:
            data = json.load(f)
        with open(new_dir / f"{name}.json", 'w') as f:
            json.dump(data, f)
            
    print("Dataset filtering complete! The new metadata size should be drastically smaller.")

if __name__ == "__main__":
    filter_dataset(
        dataroot=r"D:\TATA\data 1\v1.0-trainval03_blobs",
        original_meta_dir=r"D:\TATA\data 1\v1.0-trainval_meta\v1.0-trainval",
        new_meta_dir=r"D:\TATA\data 1\v1.0-trainval03_blobs\v1.0-trainval"
    )
