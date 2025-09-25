from argparse import ArgumentParser
from pathlib import Path
import shutil
import random
import json

import kagglehub
from kagglehub.exceptions import KaggleApiHTTPError
from tqdm import tqdm



def load_data(src: str) -> str:
    try:
        path = Path(kagglehub.dataset_download(src))
        return str(path.iterdir().__next__().absolute())
    except (ValueError, KaggleApiHTTPError, StopIteration) as e:
        raise Exception(f"Error: Invalid link for dataset {src} \n Full log: {e}")




def separate_and_save_data(
        src: str, tgt: str, per: float,
        ann_train: str, ann_val: str
    ) -> None:

    src_dir = Path(src)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Error: dataset downloaded incorrect - {src_dir} does not exist")

    tgt_dir = Path(tgt)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    for ann in (ann_train, ann_val):
        ann_path = src_dir / ann
        if not ann_path.exists():
            raise Exception(f"Error: {ann_path} does not exist")
        
        print(f"Opening {ann_path.name}...")
        with open(ann_path, 'r', encoding="utf-8") as file:
            data = json.load(file)

        n_samples = int(len(data["images"]) * per)
        sampled = random.sample(data["images"], n_samples)

        pbar = tqdm(range(len(sampled)), desc=f"Sampling {ann_path.name}")
        for i in pbar:
            img = str(sampled[i]["coco_url"][30:])
            path_to_img = src_dir / img
            if not path_to_img.is_file():
                continue

            phase_dir = tgt_dir / path_to_img.parent.name
            phase_dir.mkdir(exist_ok=True)

            path_to_img.replace(phase_dir / path_to_img.name)
        
        data["images"] = sampled
        ann_sampled = tgt_dir / ann_path.parent.name
        ann_sampled.mkdir(exist_ok=True)

        print(f"Overwriting {ann_path.name}...")
        with open(ann_sampled / ann_path.name, 'w') as file:
            json.dump(data, file)
    
    test_dir = None
    for file in src_dir.glob("*test*"):
        if file.is_dir(): test_dir = file
    
    if not test_dir:
        raise FileNotFoundError(f"Error: couldn't find test data in {src}")
    
    test_dir_tgt = tgt_dir / test_dir.name
    test_dir_tgt.mkdir(exist_ok=True)

    test_imgz = list(test_dir.glob("*.jpg"))
    n_samples = int(len(test_imgz) * per)
    sampled = random.sample(test_imgz, n_samples)

    pbar = tqdm(range(len(sampled)), desc=f"Sampling {test_dir.name}")
    for i in pbar:
        sampled[i].replace(test_dir_tgt / sampled[i].name)

    print("Removing extra data")
    shutil.rmtree(src_dir.parent)
    
    

        


if __name__ == "__main__":
    parser = ArgumentParser(description="Load and separate data")

    parser.add_argument("--url", type=str, help="Link to dataset")
    parser.add_argument("--dst", type=str, help="Path to saved data")
    parser.add_argument("--per", type=float, help="Percentage of data for keeping")
    parser.add_argument("--ann", type=str, nargs=2, help="Path to train and val annotations")

    args = parser.parse_args()

    data_path = load_data(args.url)
    if data_path:
        separate_and_save_data(data_path, args.dst, args.per, *args.ann)



