import argparse
import json
import os

# To convert YOLO box format from detection results to x, y, w, h coordinates

def YOLO_to_xywh(box):
    x1, y1, x2, y2 = box
    x = int(x1)
    y = int(y1)
    w = int(x2 - x1)
    h = int(y2 - y1)
    return x, y, w, h

# To select the bounding box with the highest confidence

def select_best_box(boxes):
    best_box = None
    best_conf = -1.0

    for box in boxes:
        conf = box["confidence"]
        if conf > best_conf:
            best_conf = conf
            best_box = box

    return best_box

def detection_to_coordinates(args):

    with open(args.predictions, "r") as f:
        predictions = json.load(f)

    coordinates = {}
    total_images = 0
    skipped_images = 0

    for item in predictions:

        image_path = item["image"]
        boxes = item["boxes"]

        if not boxes:
            skipped_images += 1
            continue

        best_box = select_best_box(boxes)

        if best_box["confidence"] < args.conf_threshold:
            skipped_images += 1
            continue

        bbox_xyxy = best_box["xyxy"]
        x, y, w, h = YOLO_to_xywh(bbox_xyxy)

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        frame_key = image_name

        coordinates[frame_key] = "{} {} {} {}".format(x, y, w, h)
        total_images += 1

    with open(args.output, "w") as f:
        for key, bbox in coordinates.items():
            f.write("{} : {}\n".format(key, bbox))

    print("Detection to coordinates completed.")
    print("Saved coordinates for {} images.".format(total_images))
    print("Skipped images: {}".format(skipped_images))
    print("Output file: {}".format(args.output))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert YOLO detection output to coordinates.txt format")

    parser.add_argument("--predictions", type=str, required=True, help="Path to YOLO JSON predictions file")
    parser.add_argument("--output", type=str, default="coordinates_detected.txt", help="Output coordinates file")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for keeping detections")

    args = parser.parse_args()

    detection_to_coordinates(args)