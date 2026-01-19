import os
import random
import shutil
import cv2

# To extract frames from videos based on ground-truth TXT files and assigns them to one of the 4 original classes using image IDs
# This function is meant to be run only ONCE to create the raw classification dataset

def extract_frames_from_videos(videos_path, raw_info_path, images_by_avg_path, output_images_path, save_coordinates = False,
                               coordinates_output_file = None):
    
    coordinates = {}
    total_frames = 0

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    scene_folders = os.listdir(videos_path)
    raw_files = os.listdir(raw_info_path)
    classes = os.listdir(images_by_avg_path)

    for c in classes:
        images_per_class_path = os.path.join(images_by_avg_path, c)
        image_ids = os.listdir(images_per_class_path)

        class_folder = os.path.join(output_images_path, c)
        os.makedirs(class_folder, exist_ok=True)

        for image_name in image_ids:
            img_id = image_name.split(".")[0]

            for raw_file in raw_files:
                raw_file_path = os.path.join(raw_info_path, raw_file)
                track = raw_file.split(".")[0] 

                frames_for_video = []

                with open(raw_file_path, "r") as f:
                    for line in f:
                        info = line.strip().split()
                        frame_number = info[0]
                        txt_img_id = info[1]
                        coords = info[2:6]

                        if txt_img_id == img_id:
                            frames_for_video.append(frame_number)
                            if save_coordinates:
                                key = "{}_{}_{}".format(track, img_id, frame_number)
                                coordinates[key] = " ".join(coords)
                                total_frames += 1

                if not frames_for_video:
                    continue

                if track not in scene_folders:
                    continue

                video_path = os.path.join(videos_path, track, "scenevideo.mp4")

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError("Could not open video: {}".format(video_path))

                for frame_number in frames_for_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
                    ret, frame = cap.read()
                    if ret:
                        out_name = "{}_{}_{}.jpg".format(track, img_id, frame_number)
                        cv2.imwrite(os.path.join(class_folder, out_name), frame)

                cap.release()

    if save_coordinates:
        if coordinates_output_file is None:
            raise ValueError("coordinates_output_file must be provided if save_coordinates=True")

        with open(coordinates_output_file, "w") as f:
            for frame_key, coords in coordinates.items():
                f.write("{} : {}\n".format(frame_key, coords))

        print("Saved coordinates for {} frames to {}".format(total_frames, coordinates_output_file))

# To create classification subset directories

def create_output_folders(base_path, subsets, class_names):
    for subset in subsets:
        for cls in class_names:
            path = os.path.join(base_path, subset, cls)
            os.makedirs(path, exist_ok=True)

# To read ID to class mapping from the ground truth TXT files from the previous work.

def read_id_file(file_path):
    id_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            img_id, class_id = line.strip().split(":")
            id_dict[img_id] = class_id
    return id_dict

# To create the split for train/val

def create_train_val_split(train_ids, val_ratio, seed=None):
    if seed is not None:
        random.seed(seed)

    ids = train_ids.copy()
    random.shuffle(ids)

    split_idx = int(len(ids) * (1 - val_ratio))
    return ids[:split_idx], ids[split_idx:]

# To move the frames to their respective directories

def move_classification_frames(original_data_path, output_data_path, original_classes, subsets, new_class_map, train_ids, val_ids,
    test_ids, train_ids_dict, test_ids_dict):

    os.makedirs(output_data_path, exist_ok=True)
    for subset in subsets:
        for cls in new_class_map.values():
            os.makedirs(os.path.join(output_data_path, subset, cls), exist_ok=True)

    total_images = 0
    for original_class in original_classes:
        class_path = os.path.join(original_data_path, original_class)
        if not os.path.exists(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_id = img_name.split("_")[1]
            src_path = os.path.join(class_path, img_name)

            if img_id in train_ids:
                dst_subset = "train"
                dst_class = new_class_map[train_ids_dict[img_id]]

            elif img_id in val_ids:
                dst_subset = "val"
                dst_class = new_class_map[train_ids_dict[img_id]]

            elif img_id in test_ids:
                dst_subset = "test"
                dst_class = new_class_map[test_ids_dict[img_id]]

            else:
                continue

            dst_path = os.path.join(output_data_path, dst_subset, dst_class, img_name)

            shutil.move(src_path, dst_path)
            total_images += 1

    return total_images