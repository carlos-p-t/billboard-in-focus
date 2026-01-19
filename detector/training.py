import argparse
import json
from ultralytics import YOLO

# Training model module

def train_model(args):

    print("Starting YOLO detection training ...\n")
    print(args)

    # Loading pre-trained model
    model = YOLO(args.model)

    # Training arguments
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "scale": args.scale,
        "plots": True,
        "name": args.name
        }

    # Optional arguments
    if args.lr0 is not None:
        train_args["lr0"] = args.lr0

    if args.optimizer is not None:
        train_args["optimizer"] = args.optimizer

    if args.patience is not None:
        train_args["patience"] = args.patience
    
    # Training through YOLO framework
    results = model.train(**train_args)    
    
    # Extracting metrics
    metrics = {
        "precision": results.results_dict["metrics/precision(B)"],
        "recall": results.results_dict["metrics/recall(B)"],
        "mAP50": results.results_dict["metrics/mAP50(B)"],
        "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
        "fitness": results.results_dict["fitness"]
        }
    
    # Displaying metrics
    print("Training metrics: \n")
    for key, val in metrics.items():
        print("{}: {}".format(key, val))

    # Saving metrics to a file
    with open("runs/detect/{}/training_metrics.json".format(args.name), "w") as f:
        json.dump(metrics, f, indent=4)

    # Evaluating the best model on test set
    print("Evaluating the best model on the test set ... \n")
    best_model = YOLO("runs/detect/{}/weights/best.pt".format(args.name))
    best_results = best_model.val(data=args.data, split="test")

    # Extracting test metrics
    test_metrics = {
        "precision": best_results.results_dict["metrics/precision(B)"],
        "recall": best_results.results_dict["metrics/recall(B)"],
        "mAP50": best_results.results_dict["metrics/mAP50(B)"],
        "mAP50-95": best_results.results_dict["metrics/mAP50-95(B)"],
        "fitness": best_results.results_dict["fitness"]
        }
    
    # Displaying test metrics
    print("Test metrics: \n")
    for key, val in test_metrics.items():
        print("{}: {}".format(key, val))

    # Saving test metrics
    with open("runs/detect/{}/test_metrics.json".format(args.name), "w") as f:
        json.dump(test_metrics, f, indent=4)

# Main program

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YOLO Detection Training Script")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)

    args = parser.parse_args()

    print("Starting training with arguments:")
    print(args)

    train_model(args)