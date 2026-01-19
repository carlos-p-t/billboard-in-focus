from billboardlamac import *

# This script is meant to be modified for reproducibility
# Configuration

VIDEOS_PATH = ".../data/eye tracker"
RAW_INFO_PATH = ".../data/tracks/ground_truth"
IDS_CLASSIFICATION = ".../data/ids for classification/image_ids"
ORIGINAL_DATA_PATH = ".../data/original_classification"
OUTPUT_DATA_PATH = ".../data/classification"
TRAIN_ID_FILE = ".../data/classification/train_id.txt"
TEST_ID_FILE = ".../data/classification/test_id.txt"
COORDINATES_PATH = ".../data/classification/gt_coordinates.txt" # Ground Truth coordinates
RUN_FRAME_EXTRACTION = True # Change to false if the script was already executed

ORIGINAL_CLASSES = ["long", "medium", "none", "short"]
NEW_CLASSES = ["long", "short", "none"]
NEW_CLASS_MAP = {"0": "long", "1": "short", "2": "none"}

SUBSETS = ["train", "val", "test"]
VAL_SPLIT_RATIO = 0.13
RANDOM_SEED = 20

def main():

    if RUN_FRAME_EXTRACTION:
        extract_frames_from_videos(VIDEOS_PATH, RAW_INFO_PATH, IDS_CLASSIFICATION, ORIGINAL_DATA_PATH, True, COORDINATES_PATH)

    train_ids_dict = read_id_file(TRAIN_ID_FILE)
    test_ids_dict = read_id_file(TEST_ID_FILE)

    train_ids = list(train_ids_dict.keys())
    test_ids = list(test_ids_dict.keys())

    if "109" not in test_ids_dict:
        test_ids_dict["109"] = "2"
        test_ids.append("109")

    train_ids, val_ids = create_train_val_split(train_ids, VAL_SPLIT_RATIO, RANDOM_SEED)

    create_output_folders(OUTPUT_DATA_PATH, SUBSETS, NEW_CLASSES)

    move_classification_frames(ORIGINAL_DATA_PATH, OUTPUT_DATA_PATH, ORIGINAL_CLASSES, SUBSETS, NEW_CLASS_MAP, train_ids,
        val_ids, test_ids, train_ids_dict, test_ids_dict)
    

if __name__ == "__main__":
    main()