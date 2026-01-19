import argparse
import json
from ultralytics import YOLO

# Evaluation model module

def evaluate_model(args):

    print("Starting YOLO model evaluation...\n")
    print(args)

    # Loading trained model
    model = YOLO(args.model)

    # Running validation
    results = model.val(data=args.data, split=args.split, device=args.device)

    # Extracting metrics
    metrics = {
        "precision": results.results_dict["metrics/precision(B)"],
        "recall": results.results_dict["metrics/recall(B)"],
        "mAP50": results.results_dict["metrics/mAP50(B)"],
        "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
        "fitness": results.results_dict["fitness"]
        }

    print("Evaluation metrics: \n")
    for key, val in metrics.items():
        print("{}: {}".format(key, val))

    # Saving metrics
    output_file = args.output
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved to: {}".format(output_file))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YOLO Model Evaluation Script")

    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO weights (best.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--device", type=int, default=0, help="GPU device id")
    parser.add_argument("--output", type=str, default="evaluation_metrics.json", help="Output JSON file to store metrics")

    args = parser.parse_args()

    evaluate_model(args)