from argparse import ArgumentParser
from pathlib import Path
import shutil

from pycocotools.coco import COCO



class YOLOFormater:

    def __init__(
            self, 
            root: str, 
            tgt: str,
            yaml: str, 
            ann_train: str,
            ann_val: str
        ):

        self.ds_path = Path(root)
        if not self.ds_path.is_dir():
            raise FileNotFoundError(f"Error: couldn't find {self.ds_path.absolute()}")
        
        self.prep_path = Path(tgt)

        self.imgz_folders = {}
        self.__prep_imgz()

        categories = {}
        self.annotated_phases = {}

        for ann in (ann_train, ann_val):
            categories.update(self.__transform_ann(ann))

        self.sorted_cats = dict(sorted(categories.items()))
        self.create_yaml(yaml)



    def __prep_imgz(self) -> None:
        imgz_path = self.prep_path  / "images"
        imgz_path.mkdir(parents=True, exist_ok=True)

        for phase in ("train", "val", "test"):
            for file in self.ds_path.glob(f"*{phase}*"):
                folder = Path(file)
                if not folder.is_dir():
                    continue

                new_folder = imgz_path / folder.name
                self.imgz_folders[phase] = new_folder

                if new_folder.is_dir():
                    shutil.rmtree(new_folder)
                shutil.copytree(folder, new_folder)



    def __transform_ann(self, ann: str) -> dict[int, str]:
        ann_path = self.ds_path / ann
        if not ann_path.is_file():
            raise FileNotFoundError(f"Error: couldn't find {ann_path.absolute()}")
        
        coco = COCO(ann_path)

        lblz_path = self.prep_path / "labels"
        lblz_path.mkdir(parents=True, exist_ok=True)

        for id in coco.imgs:
            ann_ids = coco.getAnnIds(imgIds=id)
            anns = coco.loadAnns(ann_ids)

            img_info = coco.loadImgs(id)[0]
            h, w = img_info["height"], img_info["width"]
            
            img = Path(img_info["coco_url"][30:])
            phase = lblz_path / img.parent
            phase.mkdir(exist_ok=True)
            self.annotated_phases[ann_path.stem] = phase.name

            with open(phase / f"{img.stem}.txt", 'w') as file:
                for lbl in anns:
                    if "bbox" in lbl:
                        x, y, bw, bh = lbl["bbox"]
                        cx = (x + bw / 2) / w
                        cy = (y + bh / 2) / h
                        bw /= w
                        bh /= h
                        file.write(f"{lbl['category_id']} {cx} {cy} {bw} {bh}\n")
            

        cat_ids = coco.getCatIds()
        cats = coco.loadCats(cat_ids)
        return {int(cat["id"])-1: cat["name"] for cat in cats}


    
    def create_yaml(self, yaml: str) -> None:

        lines = [
            f"path: {self.prep_path}\n",
            f"train: {self.imgz_folders["train"].parent.name}/{self.imgz_folders["train"].name}\n",
            f"val: {self.imgz_folders["val"].parent.name}/{self.imgz_folders["train"].name}\n",
            f"nc: {len(self.sorted_cats)}\n",
            f"names: {list(self.sorted_cats.values())}",
        ]

        # for id, cls in self.sorted_cats.items():
        #     lines.append(f"\t{id}: {cls}\n")

        with open(self.ds_path.parent / Path(yaml).name, 'w') as file:
            file.writelines(lines)



if __name__ == "__main__":
    parser = ArgumentParser(description="Transform data to YOLO notation")
    parser.add_argument("--src", type=str, help="Path to root of dataset")
    parser.add_argument("--dst", type=str, help="Folder for saving prepared data")
    parser.add_argument("--yml", type=str, help="Path to YOLO config")
    parser.add_argument("--ann", type=str, nargs=2, help="Path to train and val annotation files")

    args = parser.parse_args()

    YOLOFormater(args.src, args.dst, args.yml, *args.ann)    