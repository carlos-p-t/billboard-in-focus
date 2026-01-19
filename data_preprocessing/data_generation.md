# Preparing the Data

## Creating the Mapillary Vistas Subset for Detection

Main script: ```detection_data.py```

This script uses utilities from ```mapillary.py``` to:

* Select images from Mapillary Vistas that contain billboards
* Sample additional images without billboards
* Generate YOLO-format bounding box labels from panoptic annotations
* Create training, validation, and testing subsets

From the original Mapillary Vistas dataset, this script generates a subset containing the following:

* **Training set**

    * 5000 images containing billboards
    * 1000 images without billboards

* **Validation set**

    * 1000 images with billboards
    * 200 images without billboards

* **Test set**

    * 2500 images (copied from Mapillary test split)

The generated subset will be in YOLO format and saved under ```data/detection/mapillary_vistas/```. The annotations for the test subset is already placed in its respective directory.

### Running the script

First, the script must be edited to point the local Mapillary Vistas directory:

``` MAPILLARY_ROOT = "path/to/mapillary_vistas_v1/" ```

Then, the script can be executed:

``` python data_preprocessing/detection_data.py ```

This will automatically generate the subset data required for the *base training* stage.

The dataset for the *fine tuning* stage is already in YOLO format and can be downloaded with the rest of the BillboardLamac dataset from [HuggingFace](https://huggingface.co/datasets/carlospizarroso/BillboardLamac). Once downloaded, it is highly recommended to place it in the ```/data``` directory. The BillboardLamac dataset for detection should be manually placed in ```/data/detection/billboard_lamac```.

## Preparing the BillboardLamac Data for Classification

Main script: ```classification_data.py```

This script uses functions from ```billboardlamac.py``` to build the gaze classification dataset. 

The script performs the following operations:

**1. Frame extraction**

* Extracts frames from the original BillboardLamac videos using ground-truth annotation files.
* Each extracted frame is assigned to one of the original four gaze duration classes.

**2. Class reorganization**

* The four original classes are merged into three final classes:

    * long
    * short
    * none

**3. Dataset splitting**

* Reconstructs the official train/test split from provided ID files.
* Creates an additional validation split from the training data.

**4. Dataset formatting**

* Organizes the final dataset into a standard ImageFolder-style directory structure suitable for training classifiers.

### Expected input

The script requires:

* Original BillboardLamac videos
* Ground-truth tracking files
* Image ID mappings from previous work

Paths to these resources must be configured at the top of ```classification_data.py```.

### Running the script

The path to where the original BillboardLamac dataset was downloaded must be first configurated in ```classification_data.py```. 

Then, the script can be executed:

```python data_preprocessing/classification_data.py```

This will automatically generate the data for the classification task.