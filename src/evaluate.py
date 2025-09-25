import yaml
import numpy as np

from ultralytics.models import YOLO


def get_worst_class(metric, data_yaml):
    with open(data_yaml, "r") as f:
        class_names = yaml.safe_load(f)["names"]
    
    worst_idx = np.np.argsort(metric)[:3]
    print(class_names[worst_idx], metric[worst_idx], end="\n\n")


if __name__ == "__main__":
    model = YOLO("outs/exp2/weights/best.pt")
    metrics = model.val(split="val")

    yml = "data/yolo_cfg.yaml"

    get_worst_class(metrics.box.map_per_class, yml)
    get_worst_class(metrics.box.precision_per_class, yml)
    get_worst_class(metrics.box.recall_per_class, yml)


