from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import json

import pandas as pd
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


class DSInfo:
    def __init__(self, ds_root: str, dst_folder: str, ann_train: str, ann_val: str) -> None:
        root = Path(ds_root)
        DSInfo.__verificate_path(root)

        phases = ["TRAIN", "VAL", "TEST"]
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

        self.save_info(dst_folder)


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
    

    def __cls_table(self, dst: Path) -> None:
        cls_df = pd.DataFrame(self.get_classes())

        columns = ['NUM'] + cls_df.columns.to_list()
        raws = [[str(i+1)] + list(row) for i, row in zip(cls_df.index, cls_df.values)]

        fig, ax = plt.subplots(figsize=(len(columns), round(len(raws)/5.5)))
        table = ax.table(colLabels = columns, cellText = raws, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 0.8)
        ax.set_title("Class labels")
        ax.axis("off")

        fig.savefig(dst / "class_labels.jpg")
        plt.close(fig)


    def __ext_table(self, dst: Path) -> None:
        ext_df = pd.DataFrame(self.get_possible_extensions()).fillna(0).astype(int)
        ext_df.loc["TOTAL"] = self.get_images_amount()

        columns = ['EXT'] + ext_df.columns.to_list()
        raws = [[ext] + list(row) for ext, row in zip(ext_df.index, ext_df.values)]
 
        fig, ax = plt.subplots(figsize=(len(columns), len(raws)*2))
        table = ax.table(colLabels = columns, cellText = raws, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax.set_title("Extension types")
        ax.axis("off")

        fig.savefig(dst / "amount_images.jpg")
        plt.close(fig)


    def __distr_hist(self, dst: Path) -> None:
        distr_df = pd.DataFrame(self.get_distribution()).fillna(0).astype(int)
        lbls, cols = distr_df.index, distr_df.columns

        fig, axs = plt.subplots(1, 2, figsize=(len(cols)*10, round(len(lbls)/5.5)))

        for col, ax in zip(cols, axs):
            colors = ["red" if x < 1e2 else "orange" if x < 1e3 else "yellow" if x < 1e4 else "lightgreen" for x in distr_df[col]]
            ax.barh(lbls, distr_df[col], color=colors)
            ax.set_xscale('log')
            ax.set_ylabel("classes")
            ax.set_xlabel("n_imgz")
            ax.set_title(f"{col} dist")

        fig.savefig(dst / "distribution_classes.jpg")
        plt.close(fig)
            
    
    def __resol_plot(self, dst: Path) -> None:
        resols = self.get_resolutions()
        titles = resols.keys()

        fig, axs = plt.subplots(1, 2, figsize=(8, 6))
        for title, ax in zip(titles, axs):
            coords = resols[title]
            edges = {
                "minx": min(coords, key=lambda c: c[0]),
                "miny": min(coords, key=lambda c: c[1]),
                "maxx": max(coords, key=lambda c: c[0]),
                "maxy": max(coords, key=lambda c: c[1])
            }
            
            x, y = [coord[0] for coord in coords], \
                [coord[1] for coord in coords]
            ax.scatter(x, y)
            
            average_x, average_y = round(sum(x) / len(x)), \
                round(sum(y) / len(y))
            ax.scatter([average_x], [average_y], c="lightgreen", marker='o')
            ax.plot([average_x, average_x], [0, average_y], 'k--', c="lightgreen")
            ax.plot([0, average_x], [average_y, average_y], 'k--', c="lightgreen")

            for edge in edges.values():
                ex, ey = edge
                ax.scatter([ex], [ey], c="orange", marker='P')
                ax.plot([ex, ex], [0, ey], 'k--', c="orange")
                ax.plot([0, ex], [ey, ey], 'k--', c="orange")

            ax.set_ylabel("height")
            ax.set_xlabel("width")
            ax.set_title(f"Resolutions in {title}")
            
        fig.savefig(dst / "resolutions.jpg")
        plt.close(fig)
        



    def save_info(self, dst_folder: str) -> None:
        dst = Path(dst_folder)
        dst.mkdir(parents=True, exist_ok=True)

        self.__cls_table(dst)
        self.__ext_table(dst)
        self.__distr_hist(dst)
        self.__resol_plot(dst)

        
        



if __name__ == "__main__":
    parser = ArgumentParser(description="Collect and save info about dataset")
    parser.add_argument("--src", type=str, help="Path to root of dataset")
    parser.add_argument("--dst", type=str, help="Path to save result")
    parser.add_argument("--ann", type=str, nargs=2, help="Path to train and val annotations")
    args = parser.parse_args()

    DSInfo(args.src, args.dst, *args.ann)
