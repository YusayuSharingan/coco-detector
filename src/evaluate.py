from argparse import ArgumentParser
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
from torch.cuda import is_available
from ultralytics.models import YOLO
from ultralytics.utils.metrics import Metric


class Evaluating:

    def __init__(self, wght: str, data: str, use_gpu: bool, outs = "outs"):
        metrics = self.__get_metrics(wght, data, use_gpu)
        self.__cls_names = self.__get_cls(data)

        self.__plot_metric(metrics.maps, "mAP50", outs)
        # self.__plot_metric(p, 'Prescision', outs)
        # self.__plot_metric(r, 'Recall', outs)
        # self.__plot_metric(ap50, 'mAP50', outs)
        # self.__plot_metric(ap, 'mAP50-95', outs)



    @staticmethod
    def __verification(path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Error: couldn't find {p.absolute()}")
        return p


    def __get_metrics(self, weights: str, data: str, use_gpu: bool):
        Evaluating.__verification(weights)
        device = "cpu"
        if use_gpu:
            if not is_available():
                print("Warning: cuda is not avalibale")
            device = "cuda"
        model = YOLO(weights).to(device)
        return model.val(split="val", data=Evaluating.__verification(data))


    def __get_cls(self, data_yaml: str):
        with open(Evaluating.__verification(data_yaml), "r") as f:
            return yaml.safe_load(f)["names"]


    def __plot_metric(self, metric: Metric, metric_name: str, outs: str) -> None:
        fig, ax = plt.subplots(figsize=(8, round(len(self.__cls_names)/5.5)))
        ax.barh(self.__cls_names, metric)
        ax.set_title(metric_name)
        ax.set_xlabel("values")
        ax.set_ylabel("classes")

        dst = Path(outs)
        dst.mkdir(exist_ok=True)
        fig.savefig(f"{dst / metric_name}.jpg")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluationg model")
    parser.add_argument("--wght", type=str, help="Path to evaluating model")
    parser.add_argument("--data", type=str, help="Path to config")
    parser.add_argument("--gpu", action="store_true", help="Permisson to use CUDA")
    parser.add_argument("--outs", type=str, help="File for logging metrics")
    args = parser.parse_args()

    Evaluating(args.wght, args.data, args.gpu, args.outs)

