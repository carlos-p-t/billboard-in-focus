from mapillary import *

# This file is meant to be edited for reproducibility
# Configuration
# Insert Mapillary root directory
MAPILLARY_ROOT = ".../Mapillary-vistas-dataset/mapillary_vistas_v1/"

# Splits, should not be changed since Mapillary provides the three of them
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"

# Subsets Sizes (to be modified accordingly)
TRAIN_SIZE_B = 5000 # Train images with billboards
TRAIN_SIZE_NB = 1000 # Train images without billboards
VAL_SIZE_B = 1000 # Validation iamges with billboards
VAL_SIZE_NB = 200 # Validation images without billboards
TEST_SIZE = 2500 # Test images with and without billboards

# Output folders
TRAIN_IMG = "data/detection/mapillary_vistas/train/images"
TRAIN_LBL = "data/detection/mapillary_vistas/train/labels"
VAL_IMG = "data/detection/mapillary_vistas/val/images"
VAL_LBL = "data/detection/mapillary_vistas/val/labels"
TEST_IMG = "data/detection/mapillary_vistas/test/images"
TEST_LBL = "data/detection/mapillary_vistas/test/labels"   # Already provided

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_LBL, exist_ok=True)
os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)
os.makedirs(TEST_IMG, exist_ok=True)
os.makedirs(TEST_LBL, exist_ok=True)

# Training subset
train_billboards = select_billboard_images(MAPILLARY_ROOT, TRAIN_SPLIT, size=TRAIN_SIZE_B) # We selected 5000 images with billboards
train_non_billboards = select_non_billboard_images(MAPILLARY_ROOT, TRAIN_SPLIT, exclude_list=train_billboards, size=TRAIN_SIZE_NB) # And added 1000 images without billboards
train_all = train_billboards + train_non_billboards

# Copying the selected files to create a subset and creating their labels
copy_images(train_all, os.path.join(MAPILLARY_ROOT, TRAIN_SPLIT, "images"), TRAIN_IMG)
create_yolo_labels(train_all, MAPILLARY_ROOT, TRAIN_SPLIT, TRAIN_LBL)


# Validation subset
val_billboards = select_billboard_images(MAPILLARY_ROOT, VAL_SPLIT, size=VAL_SIZE_B) # we selected 1000 images with billboards
val_non_billboards = select_non_billboard_images(MAPILLARY_ROOT, VAL_SPLIT, exclude_list=val_billboards, size=VAL_SIZE_NB) # And added 200 images without bllboards
val_all = val_billboards + val_non_billboards

# Copying the selected files to create a subset and creating their labels
copy_images(val_all, os.path.join(MAPILLARY_ROOT, VAL_SPLIT, "images"), VAL_IMG)
create_yolo_labels(val_all, MAPILLARY_ROOT, VAL_SPLIT, VAL_LBL)

# Testig subset (Mapillary testing has NO labels, so we just copy images)

testing_images = os.listdir(os.path.join(MAPILLARY_ROOT, TEST_SPLIT, "images"))

# Remove extensions to get IDs
testing_images = [f.split(".")[0] for f in testing_images]

# Take first 2500 images (no filtering)
test_selected = testing_images[:TEST_SIZE]

# Copying selected images
copy_images(test_selected, os.path.join(MAPILLARY_ROOT, TEST_SPLIT, "images"), TEST_IMG)

print("Dataset creation finished!")
print("Training images:", len(train_all))
print("Validation images:", len(val_all))
print("Testing images:", len(test_selected))