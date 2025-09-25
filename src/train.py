from argparse import ArgumentParser
from pathlib import Path

from torch.cuda import is_available
from ultralytics.models import YOLO



def train_yolo(
        data_yaml: str,
        weights="yolo11n.pt",
        use_gpu=False,
        opt="AdamW",
        epochs=3,
        batch=16,
        imgsz=640,
        lr0=0.1,
        project="outs"
        ):

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Error: couldn't find {data_yaml}")

    device = "cpu"
    if use_gpu:
        if not is_available():
            print("Warning: cuda is not avalibale")
        device = "cuda"

    model = YOLO(weights, task='detect').to(device)

    return model.train(
        data=data_yaml,
        optimizer=opt,
        project=project,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0
    )



if __name__ == "__main__":
    parser = ArgumentParser(description="Train stage of the pipeline")
    parser.add_argument("--data", type=str, help="Path to config")
    parser.add_argument("--wght", type=str, default="yolo11n.pt", help="Path to weights of YOLO")
    parser.add_argument("--gpu", action="store_true", help="Permisson to use CUDA")
    parser.add_argument("--opt", type=str, default="AdamW", choices=["Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"], help="Optimizer type")
    parser.add_argument("--epo", type=int, default=3, help="Num of training epochs")
    parser.add_argument("--bsz", type=int, default=16, help="Batch size")
    parser.add_argument("--isz", type=int,  default=640, help="Size of input images")
    parser.add_argument("--lr0", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--pjct", type=str, default="outs", help="Path to save outs of experiments")

    args = parser.parse_args()

    train_yolo(args.data, args.wght, args.gpu, args.opt, args.epo, args.bsz, args.isz, args.lr0, args.pjct)
