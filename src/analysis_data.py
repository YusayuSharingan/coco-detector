from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
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
            for file in root.glob(f"*{phase.lower()}*"):
                if file.is_dir(): self.__ds_pathes[phase] = file

            if not self.__ds_pathes[phase]: 
                raise FileNotFoundError(f"Error: couldn't find images for {phase} in {root.absolute()}")
        
        
        self.__pos_exts = {p: self.__cnt_exts(p) for p in phases}
        self.__imgz_amount = {p: sum(self.__pos_exts[p].values()) for p in phases}


        self.__cls_ids = {}
        self.__disrtibution = {}
        self.__bbox_distribution = {}
        self.__resolutions = {}
        for ann, phase in zip((ann_train, ann_val), phases[:2]):
            ann_path = root / ann
            DSInfo.__verificate_path(ann_path)

            coco = COCO(ann_path)

            self.__cls_ids[phase] = self.__get_cls(coco)

            cls_distr_by_ids = self.__cnt_cls(coco)
            self.__disrtibution[phase] = {self.__cls_ids[phase][idx]: n for idx, n in cls_distr_by_ids.items()}
            
            bbox_distr_by_ids = self.__cnt_bbox(coco)
            self.__bbox_distribution[phase] = {self.__cls_ids[phase][idx]: boxsz for idx, boxsz in bbox_distr_by_ids.items()}
            self.__resolutions[phase] = self.__get_resols(coco)

        self.save_info(dst_folder)


    @staticmethod
    def __verificate_path(file: Path):
        if not file.exists(): raise FileNotFoundError(f"Error: {file.absolute()} does not exist")


    def __get_cls(self, coco: COCO) -> dict[str, dict[int, str]]:
        cat_ids = coco.getCatIds()
        cats = coco.loadCats(cat_ids)
        return {int(cat["id"]): cat["name"] for cat in cats}


    def __cnt_cls(self, coco: COCO) -> dict[int, int]:
        counter = defaultdict(int)

        for id in coco.imgs:
            ann_ids = coco.getAnnIds(imgIds=id)
            annotations = coco.loadAnns(ann_ids)
            for ann in annotations:
                counter[ann["category_id"]] += 1
        
        return dict(counter)


    def __cnt_bbox(self, coco: COCO) -> dict[int, tuple[int, int]]:
        bbox_w = defaultdict(float)
        bbox_h = defaultdict(float)
        counter = defaultdict(int)

        for id in coco.imgs:
            annIds = coco.getAnnIds(imgIds=id)
            anns = coco.loadAnns(annIds)

            imgInfo = coco.loadImgs(id)[0]
            h, w = imgInfo["height"], imgInfo["width"]

            for ann in anns:
                _, _, bw, bh = ann["bbox"]
                bbox_w[ann["category_id"]] += bw/w
                bbox_h[ann["category_id"]] += bh/h        
                counter[ann["category_id"]] += 1
                
        return {idx: (bbox_w[idx] / counter[idx], bbox_h[idx] / counter[idx]) for idx in counter.keys()}


    def __get_resols(self, coco: COCO) -> list[tuple[int, int]]:
        resols = []
        
        for id in coco.imgs:
            imgInfo = coco.loadImgs(id)[0]
            resols.append((imgInfo["width"], imgInfo["height"]))    
        
        return resols


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
    
    def get_bbox_distribution(self) -> dict[str, dict[str, tuple[int, int]]]:
        return self.__bbox_distribution

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
            ax.plot([average_x, average_x], [0, average_y], '--', c="lightgreen")
            ax.plot([0, average_x], [average_y, average_y], '--', c="lightgreen")

            for edge in edges.values():
                ex, ey = edge
                ax.scatter([ex], [ey], c="orange", marker='P')
                ax.plot([ex, ex], [0, ey], '--', c="orange")
                ax.plot([0, ex], [ey, ey], '--', c="orange")

            ax.set_ylabel("height")
            ax.set_xlabel("width")
            ax.set_title(f"Resolutions in {title}")
            
        fig.savefig(dst / "resolutions.jpg")
        plt.close(fig)

    def __bbox_distr_hist(self, dst: Path) -> None:        
        bbox_distr_df = pd.DataFrame(self.get_bbox_distribution())
        cols, lbls = bbox_distr_df.columns, bbox_distr_df.index
        
        col_width = 0.35
        y = np.arange(len(lbls))

        fig, axs = plt.subplots(1, 2, figsize=(len(cols)*10, round(len(lbls)/5.5)))
        for col, ax in zip(cols, axs):
            widths = bbox_distr_df[col].map(lambda x: x[0])
            height = bbox_distr_df[col].map(lambda x: x[1])

            ax.barh(y - col_width/2, widths, col_width, label="width")
            ax.barh(y + col_width/2, height, col_width, label="height")
            ax.legend()
            ax.set_yticks(y, lbls)
            ax.set_title(f"{col} dist")
        
        fig.savefig(dst / "distribution_bboxes.jpg")
        plt.close(fig)
        


    def save_info(self, dst_folder: str) -> None:
        dst = Path(dst_folder)
        dst.mkdir(parents=True, exist_ok=True)

        self.__cls_table(dst)
        self.__ext_table(dst)
        self.__distr_hist(dst)
        self.__bbox_distr_hist(dst)
        self.__resol_plot(dst)

        
        



if __name__ == "__main__":
    parser = ArgumentParser(description="Collect and save info about dataset")
    parser.add_argument("--src", type=str, help="Path to root of dataset")
    parser.add_argument("--dst", type=str, help="Path to save result")
    parser.add_argument("--ann", type=str, nargs=2, help="Path to train and val annotations")
    args = parser.parse_args()

    DSInfo(args.src, args.dst, *args.ann)
