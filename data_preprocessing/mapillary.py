import json
import os
import shutil
import numpy as np
from PIL import Image

# Utility loaders
# Loads config file provided by Mapillary Vistas, returns a dictionary mapping label names to label IDs.

def load_config(mapillary_root):
    with open(os.path.join(mapillary_root, "config.json")) as config_file:
        config = json.load(config_file)

    labels = config["labels"]
    label_dic = {}

    for label_id, label in enumerate(labels):
        label_dic[label["name"]] = label_id

    return label_dic

# Loads panoptic information, which contains annotations given by Mapillary Vistas. Returns annotations and categories dictionaries.

def load_panoptic(mapillary_root, split="training"):
    panoptic_file = os.path.join(mapillary_root, split, "panoptic", "panoptic_2018.json")

    with open(panoptic_file) as panoptic_file:
        panoptic = json.load(panoptic_file)

    # Converts annotation information to image_id indexed dictionary
    panoptic_per_image_id = {}
    for annotation in panoptic["annotations"]:
        panoptic_per_image_id[annotation["image_id"]] = annotation

    # Converts category information to image_id indexed dictionary
    panoptic_category_per_id = {}
    for category in panoptic["categories"]:
        panoptic_category_per_id[category["id"]] = category

    return panoptic_per_image_id, panoptic_category_per_id

# Selecting images which contains Billboards
# To check billboard presence in images

def image_contains_billboard(image_id, panoptic_per_image_id, panoptic_category_per_id, panoptic_path, label_dic):

    panoptic_image = Image.open(os.path.join(panoptic_path, "{}.png".format(image_id)))
    panoptic_array = np.array(panoptic_image).astype(np.uint32)
    panoptic_id_array = (panoptic_array[:, :, 0] + (2 ** 8) * panoptic_array[:, :, 1] + (2 ** 16) * panoptic_array[:, :, 2])
    example_panoptic = panoptic_per_image_id[image_id]
    example_segments = {}

    for segment_info in example_panoptic["segments_info"]:
        example_segments[segment_info["id"]] = segment_info

    panoptic_ids_from_image = np.unique(panoptic_id_array)

    for panoptic_id in panoptic_ids_from_image:
        if panoptic_id == 0:
            continue

        if panoptic_id not in example_segments:
            continue

        segment_info = example_segments[panoptic_id]
        category = panoptic_category_per_id[segment_info["category_id"]]

        img_label = label_dic[category["supercategory"]]

        if img_label == 35:  # billboard label from Mapillary Vistas
            return True

        example_segments.pop(panoptic_id)

    return False

# To select billboard image IDs and create the subset of Mapillary Vistas

def select_billboard_images(mapillary_root, split, size=None):

    label_dic = load_config(mapillary_root)
    panoptic_per_image_id, panoptic_category_per_id = load_panoptic(mapillary_root, split)

    images_path = os.path.join(mapillary_root, split, "images")
    panoptic_path = os.path.join(mapillary_root, split, "panoptic")

    image_filenames = os.listdir(images_path)
    subset_names = []

    i = 0
    while size is None or len(subset_names) < size:
        if i >= len(image_filenames):
            break

        filename = image_filenames[i]
        i += 1

        if not filename.endswith(".jpg"):
            continue

        image_id = filename[: filename.find(".")]

        if image_id not in panoptic_per_image_id:
            continue

        has_billboard = image_contains_billboard(image_id, panoptic_per_image_id, panoptic_category_per_id, panoptic_path, label_dic)

        if has_billboard:
            subset_names.append(image_id)
            print(image_id)

    return subset_names

# To select images that do NOT contain billboards

def select_non_billboard_images(mapillary_root, split, exclude_list, size):

    label_dic = load_config(mapillary_root)
    panoptic_per_image_id, panoptic_category_per_id = load_panoptic(mapillary_root, split)

    images_path = os.path.join(mapillary_root, split, "images")
    panoptic_path = os.path.join(mapillary_root, split, "panoptic")

    image_filenames = os.listdir(images_path)
    nobillboard_names = []

    i = 0
    while len(nobillboard_names) < size:
        if i >= len(image_filenames):
            break

        filename = image_filenames[i]
        i += 1

        if not filename.endswith(".jpg"):
            continue

        image_id = filename[: filename.find(".")]

        if image_id in exclude_list:
            continue

        if image_id not in panoptic_per_image_id:
            continue

        has_billboard = image_contains_billboard(image_id, panoptic_per_image_id, panoptic_category_per_id, panoptic_path, label_dic)

        if not has_billboard:
            nobillboard_names.append(image_id)
            print(image_id)

    return nobillboard_names

# Dataset creation utilities
# To save image names in a text file

def save_subset_names(subset_names, output_file):
    with open(output_file, "w") as f:
        for image_id in subset_names:
            f.write("{}\n".format(image_id))

# To copy images in case of creating a subset of Mapillary Vistas

def copy_images(subset_names, source_path, destination_path):
    os.makedirs(destination_path, exist_ok=True)

    for image_id in subset_names:
        source = os.path.join(source_path, "{}.jpg".format(image_id))
        destination = os.path.join(destination_path, "{}.jpg".format(image_id))
        shutil.copyfile(source, destination)

# To load the text file where image names are stored

def load_subset_names(input_file):
    subset_names = []
    with open(input_file, "r") as f:
        for line in f:
            subset_names.append(line.strip())
    return subset_names

# YOLO label creation
# To create YOLO labels by extracting billboard bounding boxes from panopotic annotations

def create_yolo_labels(subset_names, mapillary_root, split, labels_output_path):

    label_dic = load_config(mapillary_root)
    panoptic_per_image_id, panoptic_category_per_id = load_panoptic(mapillary_root, split)

    images_path = os.path.join(mapillary_root, split, "images")
    panoptic_path = os.path.join(mapillary_root, split, "panoptic")

    os.makedirs(labels_output_path, exist_ok=True)

    for image_id in subset_names:
        image_path = os.path.join(images_path, "{}.jpg".format(image_id))
        panoptic_image_path = os.path.join(panoptic_path, "{}.png".format(image_id))

        img = Image.open(image_path)
        img_width, img_height = img.size

        panoptic_image = Image.open(panoptic_image_path)
        panoptic_array = np.array(panoptic_image).astype(np.uint32)

        panoptic_id_array = (panoptic_array[:, :, 0] + (2**8) * panoptic_array[:, :, 1] + (2**16) * panoptic_array[:, :, 2])

        example_panoptic = panoptic_per_image_id[image_id]
        example_segments = {}

        for segment_info in example_panoptic["segments_info"]:
            example_segments[segment_info["id"]] = segment_info

        label_doc = open(os.path.join(labels_output_path, "{}.txt".format(image_id)), "w")

        panoptic_ids_from_image = np.unique(panoptic_id_array)

        for panoptic_id in panoptic_ids_from_image:
            if panoptic_id == 0:
                continue

            if panoptic_id not in example_segments:
                continue

            segment_info = example_segments[panoptic_id]
            category = panoptic_category_per_id[segment_info["category_id"]]

            img_label = label_dic[category["supercategory"]]

            if img_label != 35:
                continue

            box_x_left = segment_info["bbox"][0]
            box_y_top = segment_info["bbox"][1]
            box_width = segment_info["bbox"][2]
            box_height = segment_info["bbox"][3]

            x_center = (box_x_left + box_width / 2) / img_width
            y_center = (box_y_top + box_height / 2) / img_height
            width = box_width / img_width
            height = box_height / img_height

            label_doc.write("0 {} {} {} {}\n".format(x_center, y_center, width, height))

        label_doc.close()
        img.close()
