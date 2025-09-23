from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import json

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

class DSInfo:
    def __init__(self, ds_root: str, dst_folder: str, ann_train: str, ann_val: str) -> None:
        root = Path(ds_root)
        DSInfo.__verificate_path(root)

        phases = ["train", "val", "test"]
        self.__ds_pathes: dict[str, Path] = {p: None for p in phases}

        for phase in phases:
            for file in root.glob(f"*{phase}*"):
                if file.is_dir(): self.__ds_pathes[phase] = file

            if not self.__ds_pathes[phase]: 
                raise FileNotFoundError(f"Error: couldn't find images for {phase} in {root.absolute}")
        
        
        self.__pos_exts = {p: self.__cnt_exts(p) for p in phases}
        self.__imgz_amount = {p: sum(self.__pos_exts[p].values()) for p in phases}


        self.__cls_ids = {}
        self.__disrtibution = {}
        self.__resolutions = {}
        for ann, phase in zip((ann_train, ann_val), phases[:2]):
            ann_path = root / ann
            DSInfo.__verificate_path(ann_path)

            imgz = self.__open_annotation(ann_path)
            coco = COCO(ann_path)

            self.__cls_ids[phase] = self.__get_cls(coco)

            distr_by_ids = self.__cnt_cls(coco, imgz)
            self.__disrtibution[phase] = {self.__cls_ids[phase][idx]: n for idx, n in distr_by_ids.items()}
            self.__resolutions[phase] = self.__get_resols(imgz)

        self.save_info(dst_folder, phases)




    @staticmethod
    def __verificate_path(file: Path):
        if not file.exists(): raise FileNotFoundError(f"Error: {file.absolute()} does not exist")


    def __open_annotation(self, ann: Path):
        print(f"Opening {ann}")
        with open(ann, 'r', encoding='utf-8') as file:
            return json.load(file)["images"]


    def __get_cls(self, coco: COCO) -> dict[str, dict[str, int]]:
        cat_ids = coco.getCatIds()
        cats = coco.loadCats(cat_ids)
        return {cat["id"]: cat["name"] for cat in cats}


    def __cnt_cls(self, coco: COCO, imgz) -> dict[int, int]:
        counter = defaultdict(int)

        for img in imgz:
            ann_ids = coco.getAnnIds(imgIds=img["id"])
            annotations = coco.loadAnns(ann_ids)
            for ann in annotations:
                counter[ann["category_id"]] += 1
        
        return dict(counter)
        

    def __get_resols(self, imgz) -> list[tuple[int, int]]:
        return [(img["width"], img["height"]) for img in imgz]


    def __cnt_exts(self, phase: str) -> dict[str, int]:
        path = self.__ds_pathes[phase]
        
        print(f"Check file extensions in {path.name}")

        counter = defaultdict(int)
        for img in path.iterdir():
            counter[img.suffix] += 1

        return dict(counter)


    def get_classes(self) -> dict[str, list[str]]:
        return {p: list(c.values()) for p, c in self.__cls_ids.items()}
    
    def get_possible_extensions(self) -> dict[str, dict[str, int]]:
        return self.__pos_exts
    
    def get_images_amount(self) -> dict[str, int]:
        return self.__imgz_amount
    
    def get_distribution(self) -> dict[str, dict[str, int]]:
        return self.__disrtibution

    def get_resolutions(self) -> dict[str, list[tuple[int, int]]]:
        return self.__resolutions
    

    def __plot_cls_table(self, ax: Axes, phases: list[str]):
        classes = list(self.get_classes().values())
        n_cls = len(classes[0])

        ax.axis("off")
        table = ax.table(
            colLabels = ['#'] + phases[:2],
            cellText = list(map(tuple, zip(range(1, n_cls+1), *classes))),
            cellLoc="center",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(0.5, 0.8)

    def save_info(self, dst_folder: str, phases: list[str]) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(6, 12))

        
        self.__plot_cls_table(ax, phases)


        dst = Path(dst_folder)
        dst.mkdir(exist_ok=True)
        fig.savefig(dst / "tables.jpg")



if __name__ == "__main__":
    parser = ArgumentParser(description="Collect and save info about dataset")
    parser.add_argument("--src", type=str, help="Path to root of dataset")
    parser.add_argument("--dst", type=str, help="Path to save result")
    parser.add_argument("--ann", type=str, nargs=2, help="Path to train and val annotations")
    args = parser.parse_args()

    DSInfo(args.src, args.dst, *args.ann)
