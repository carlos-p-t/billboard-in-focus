import argparse
import json
import os
from ultralytics import YOLO


def run_inference(args):

    print("Starting YOLO inference...\n")
    print(args)

    # Loading trained model
    model = YOLO(args.model)

    # Creating output directory
    os.makedirs(args.output, exist_ok=True)

    # Running predictions
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=False, # Change to True in case images with prediction bounding boxes are needed
        save_txt=args.save_txt,
        project=args.output,
        name=args.run_name,
        device=args.device,
        stream=True
        )

    # Saving predictions as JSON
    if args.save_json:

        json_path = os.path.join(args.output, "{}_predictions.json".format(args.run_name))

        with open(json_path, "w") as f:

            f.write("[\n")
            first = True
            for result in results:

                image_data = {"image": result.path, "boxes": []}

                if result.boxes is not None and len(result.boxes) > 0:

                    if args.best_only:

                        best_idx = result.boxes.conf.argmax()
                        best_box = result.boxes[best_idx]
                        box_info = {"xyxy": best_box.xyxy.tolist()[0], "confidence": float(best_box.conf), "class": int(best_box.cls)}
                        image_data["boxes"].append(box_info)

                    else:
                        for box in result.boxes:
                            box_info = {"xyxy": box.xyxy.tolist()[0], "confidence": float(box.conf), "class": int(box.cls)}
                            image_data["boxes"].append(box_info)

                if not first:
                    f.write(",\n")
                first = False

                json.dump(image_data, f)

            f.write("\n]\n")

        print("Predictions saved to: {}".format(json_path))

    print("Inference completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YOLO Inference Script")

    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO weights")
    parser.add_argument("--source", type=str, required=True, help="Path to image or folder with images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=int, default=0, help="GPU device id")
    parser.add_argument("--output", type=str, default="inference_results", help="Directory to save outputs")
    parser.add_argument("--run_name", type=str, default="exp", help="Name of inference run")
    parser.add_argument("--save_txt", action="store_true", help="Save YOLO txt format predictions")
    parser.add_argument("--save_json", action="store_true", help="Save predictions in JSON format")
    parser.add_argument("--best_only", action="store_true", help="Save only the highest confidence bounding box per image in JSON")

    args = parser.parse_args()

    run_inference(args)