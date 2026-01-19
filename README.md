# Billboard in Focus: Estimating Driver Gaze Duration from a Single Image

This repository contains the code for the paper "Billboard in Focus: Estimating Driver Gaze Duration from a Single Image" to be presented as an oral presentation at VISAPP 2026 (arXiv: [2601.07073](https://arxiv.org/abs/2601.07073))

## Datasets

Our project uses two main public datasets. This repository does not redistribute the original data due to licensing restrictions. Users must download the datasets separately and generate the required subsets using the provided scripts.

### Mapillary Vistas 

* Publicly available at their official [website](https://www.mapillary.com/dataset/vistas). 
* Version used: v1.2. 

A subset of this dataset was used to perform the **base training** of the YOLO billboard detector. Since the original dataset contains many object categories, only a subset of images containing billboards was selected.

**Important Notes:** 

* Mapillary Vistas provides labels only for the *training* and *validation* splits.
* The *test* split does not include labels. For this project, a labeled test subset was manually created and is included in this repository.

### BillboardLamac

* Now available at [HuggingFace](https://huggingface.co/datasets/carlospizarroso/BillboardLamac).

BillboardLamac was used for two tasks:
1. **Object detection fine-tuning**:

* A dedicated subset of BillboardLamac, already formatted in YOLO format, is used to fine-tune the detector after base training.
* This subset requires no additional preprocessing.

2. **Gaze classification dataset generation:**

* Raw eye-tracking videos and ground-truth files are used to extract individual frames.
* These frames form the dataset for the gaze duration classifier.

### Data Preparation Workflow

The repository includes scripts to automatically generate all required training data.

All data generation-related scripts are located in the ```data_processing/``` directory. Refer to [this](data_preprocessing/data_generation.md) file for more details.

## Object Detector Training

The billboard detection module is based on the **Ultralytics YOLO** framework and is trained in two stages:

**1. Base training** on a subset of Mapillary Vistas </br>
**2. Fine-tuning** on the BillboardLamac detection subset

All training, evaluation, and inference operations are performed using the scripts in the ```detector/``` directory.

### Stage 1: Base Training on Mapillary Vistas

After generating the Mapillary subset using ```data_processing/detection_data.py```, the detector can be trained using the ```detector/data.yaml``` configuration file. This file defines the paths to the training, validation, and testing sets created in the Data Preparation stage.

The following command can be executed to reproduce, for instance, our YOLOv8 based model:

```python training.py --model yolov8l.pt --epochs 100 --imgsz 1280 --batch 4 --flipud 0.5 --fliplr 0.5 --scale 0.5 --name basetrainv8 --data data.yaml```

Key arguments:

* ```--model```: starting YOLO pretrained model (e.g., yolov8l.pt, yolov11l.pt)
* ```--data```: dataset configuration YAML
* ```--epochs```: number of training epochs
* ```--name```: experiment name (results will be stored under ```runs/detect/<name>/```)

The script automatically:

* Trains the model
* Prints final training metrics
* Saves metrics to JSON
* Evaluates the best model on the test set

### Stage 2: Fine-Tuning on BillboardLamac

Once the base training is completed, the best model can be fine-tuned using the dedicated BillboardLamac detection subset using the provided configuration: ```detector/lamac.yaml```.

For instance, using the YOLOv8 from the previous example, the following command can be executed to fine-tune it:

``` python training.py --model .../runs/detect/<name>/weights/best.pt --epochs 100 --imgsz 1280 --batch 6 --name finetune_v8 --data lamac.yaml```

This stage adapts the detector specifically to billboard imagery and yields the final detection model used in the paper.

### Detection Model Evaluation

Trained models can be evaluated independently using: ```detector/evaluation.py```

For instance:

```python evaluation.py --model .../runs/detect/finetune_v8/weights/best.pt --data lamac.yaml --split test --output ftv8_eval.json```

This script computes:

* Precision
* Recall
* mAP@50
* mAP@50-95
* Fitness score

and saves them to a JSON file for further analysis.

### Running Inference on Detection Models

The trained detector can be used to predict billboards on new images or folders of images using: ```detector/inference.py```

For example:

```python inference.py --model .../runs/detect/finetune_v8/weights/best.pt --source ".../any_data/**/*" --conf 0.25 --output full_dataset_inference --run_name all_data --save_json```

**Available options:**

* ```--save_txt```: save predictions in YOLO text format
* ```--save_json```: export bounding boxes and confidences as JSON
* ```--best_only```: keep only the highest-confidence detection per image

The JSON output is used as input for the gaze classification stage of the pipeline.

Two dataset configuration files are provided: ```data.yaml``` and ```lamac.yaml```. These YAML files define dataset paths and class mappings required by the YOLO framework.

### Best Models

The repository includes the best-performing models reported in the paper inside: ```detector/best_models/```. These weights allow direct reproduction of the reported results without retraining.

## Gaze Classification Training

The second stage of the pipeline focuses on estimating driver attention toward detected billboards by classifying gaze duration into three predefined categories: *none*, *short*, and *long*. This stage operates on image regions corresponding to billboard detections produced by the previous module and relies on feature extraction and machine learning classification.

The gaze classification pipeline is implemented through three main scripts located in the ```classifier/``` directory:

1. ```coordinates.py```: converts the raw YOLO detection outputs into a standardized coordinate format. It reads a JSON file containing detection predictions, selects the bounding box with the highest confidence for each image, filters detections below a configurable confidence threshold, and outputs a text file mapping each image to its corresponding billboard coordinates in $(x,y,w,h)$ format. This file serves as the input for the subsequent feature extraction step.

2. ```extract_dino_features.py```: generates visual descriptors for each detected billboard region using the DINOv2 vision transformer. For every image, features are extracted from both the full frame and the cropped billboard region. A PCA transformation is learned on the training and validation sets to reduce feature dimensionality. The script combines these visual features with additional metadata from existing CSV annotations and produces two output files:

    * ```all_features_N.csv```, containing features for the training, validation, and test sets, and
    * ```gsv_features_N.csv```, containing features for external Google Street View data.

    These CSV files constitute the final input for classifier training.

3. ```train_rfc.py```: trains and evaluates a gaze classification model using the extracted features. It employs the FLAML AutoML framework to automatically search for the best classifier and hyperparameters. A custom cross-validation strategy (RefKFold) is implemented to ensure that frames from the same instance are not split across training and testing folds. The script evaluates multiple feature combinations, reports per-frame and per-instance performance metrics, and optionally tests the trained model on the external GSV dataset.

## Pipeline

*In development*

## Citations