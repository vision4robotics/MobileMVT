"""Training entry point for the TConvNet detector of MobileMVT.

TConvNet is the geometry-aware detection backbone of MobileMVT. It is trained as
a standard single-class ("vessel") object detector on the VesselMOT detection
split using the (modified) Ultralytics training loop, then paired with the UAVC
tracker at inference time (see ``demo.py``).

Example:
    # Train from scratch with the paper's K=3 configuration
    python train.py \\
        --model ultralytics/cfg/models/tconv/tconv3win-r18-320.yaml \\
        --data  vesselmot-detection.yaml \\
        --epochs 200 --imgsz 320 --batch 32 --device 0 \\
        --name tconv-w3-r18-320-vesselmot

    # Fine-tune from a pretrained checkpoint
    python train.py --model ultralytics/cfg/models/tconv/tconv3win-r18-320.yaml \\
        --data vesselmot-detection.yaml --pretrained weights/vesselmot.pt

The default hyper-parameters match those reported in the paper: AdamW with an
initial learning rate of 1e-4 and weight decay of 5e-4, batch size 32, 200
epochs, and a 320x320 input resolution.
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TConvNet detector for MobileMVT.")
    parser.add_argument(
        "--model", type=str,
        default="ultralytics/cfg/models/tconv/tconv3win-r18-320.yaml",
        help="Model config YAML (K=3 is the paper setting).",
    )
    parser.add_argument(
        "--data", type=str, default="vesselmot-detection.yaml",
        help="Dataset config YAML (see vesselmot-detection.yaml template).",
    )
    parser.add_argument(
        "--pretrained", type=str, default="",
        help="Optional checkpoint to initialize weights from (e.g. weights/vesselmot.pt).",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=320, help="Training image size.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id(s) or 'cpu'.")
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Optimizer weight decay.")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer.")
    parser.add_argument("--name", type=str, default="tconv-w3-r18-320-vesselmot",
                        help="Run name (results saved under runs/detect/<name>).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build the model from the config; optionally load pretrained weights.
    model = YOLO(args.model, task="detect")
    if args.pretrained:
        model = model.load(args.pretrained)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        name=args.name,
    )


if __name__ == "__main__":
    main()
