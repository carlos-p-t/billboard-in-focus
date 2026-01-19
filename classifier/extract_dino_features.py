import argparse
import os

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import pipeline
from transformers.image_utils import load_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--best_frames', type=int, default=10)
    parser.add_argument('dataset_path')
    parser.add_argument('dataset_path_gsv')

    return parser.parse_args()



class BillBoardDataset():
    def __init__(self, extractor, path, split, best_frames=10):
        self.extractor = extractor
        self.path = path
        self.split = split
        self.best_frames = best_frames

        self.class_labels = ['none', 'short', 'long']

        with open(os.path.join(self.path, f'{split}_id.txt'), 'r') as f:
            class_lines = f.readlines()

        self.classes = {}
        for line in class_lines:
            instance_id = int(line.split(':')[0].strip())
            class_id = int(line.split(':')[1].strip())
            self.classes[instance_id] = class_id

        coords_file = os.path.join(path, 'coordinates.txt')
        with open(coords_file, 'r') as f:
            lines = f.readlines()

        self.data = []

        no_image = 0

        for line in lines:
            name, bbox = line.strip().split(' : ')
            person, instance_id, frame_id = name.split('_')

            instance_id = int(instance_id)
            frame_id = int(frame_id)

            if instance_id not in self.classes.keys():
                continue

            bbox = [int(x) for x in bbox.split(' ')]
            class_id = self.classes[instance_id]

            filename = os.path.join(self.path, split, self.class_labels[class_id],  f'{name}.jpg')

            if not os.path.exists(filename):
                no_image += 1
                continue

            self.data.append({'filename': filename, 'bbox': bbox, 'person': person, 'instance_id': instance_id,
                              'frame_id': frame_id, 'class_id': class_id})


        persons = list(set(x['person'] for x in self.data))
        instances = list(set(x['instance_id'] for x in self.data))

        idx_dict = {instance: {person: [] for person in persons} for instance in instances}

        for i, entry in enumerate(self.data):
            idx_dict[entry['instance_id']][entry['person']].append(i)

        if self.best_frames is not None:
            self.new_data = []

            for person in persons:
                for instance in instances:
                    idxs = idx_dict[instance][person]
                    subentries = [self.data[i] for i in idxs]
                    dims = [-x['bbox'][2] * x['bbox'][3] for x in subentries]
                    if len(dims) > self.best_frames:
                        top_idxs = np.argpartition(dims, self.best_frames)[:self.best_frames]
                        self.new_data.extend([subentries[i] for i in top_idxs])
                    else:
                        self.new_data.extend(subentries)
            self.data = self.new_data

        print(f"Loaded data: {len(self.data)} samples")

    def extract_features(self):
        for entry in tqdm(self.data):

            filename = entry['filename']

            bbox = np.array(entry['bbox'])

            image = load_image(filename)
            image_features = self.extractor(image)[0][0]

            crop = image.crop([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1]+ bbox[3]])
            # image = resize(image, [self.height, self.width]) / 255.0
            crop_features = self.extractor(crop)[0][0]

            entry['image_feature'] = np.array(image_features)
            entry['crop_feature'] = np.array(crop_features)

    def apply_pca(self, full_pca, crop_pca):
        for entry in self.data:
            entry['image_pca'] = full_pca.transform(entry['image_feature'].reshape(1, -1))[0]
            entry['crop_pca'] = crop_pca.transform(entry['crop_feature'].reshape(1, -1))[0]



def extract_dino_features(args):
    dataset_path = args.dataset_path

    original_data = np.genfromtxt('data/features_by_mean_v4.csv', delimiter=',', dtype=int, skip_header=1)
    feature_extractor = pipeline("image-feature-extraction", model="facebook/dinov2-small")

    gsv_dataset = BillBoardDataset(feature_extractor, args.dataset_path_gsv, 'test', best_frames=100)
    gsv_dataset.extract_features()

    train_dataset = BillBoardDataset(feature_extractor, args.dataset_path, 'train', best_frames=args.best_frames)
    train_dataset.extract_features()
    val_dataset = BillBoardDataset(feature_extractor, args.dataset_path, 'val', best_frames=args.best_frames)
    val_dataset.extract_features()

    full_feats = [entry['image_feature'] for entry in train_dataset.data]
    full_feats.extend([entry['image_feature'] for entry in val_dataset.data])

    full_pca = PCA()
    full_pca.fit(np.array(full_feats))

    crop_feats = [entry['crop_feature'] for entry in train_dataset.data]
    crop_feats.extend([entry['crop_feature'] for entry in val_dataset.data])

    crop_pca = PCA()
    crop_pca.fit(np.array(full_feats))

    test_dataset = BillBoardDataset(feature_extractor, args.dataset_path, 'test', best_frames=args.best_frames)
    test_dataset.extract_features()

    train_dataset.apply_pca(full_pca, crop_pca)
    val_dataset.apply_pca(full_pca, crop_pca)
    test_dataset.apply_pca(full_pca, crop_pca)
    gsv_dataset.apply_pca(full_pca, crop_pca)

    all_data = train_dataset.data
    all_data.extend(val_dataset.data)
    all_data.extend(test_dataset.data)

    with open(f'data/all_features_{args.best_frames}.csv', 'w') as f:
        f.write('# instance_id, person, frame_id, side, duration, distance, size_ratio, larger_size_ratio, mask_saliency, ratio_saliency')
        f.write(', bbox_center_x, bbox_width, bbox_center_y, bbox_height')
        f.write(f', {len(all_data[0]["image_pca"])}x image_pca')
        f.write(f', {len(all_data[0]["crop_pca"])}x crop_pca')
        f.write(f', {len(all_data[0]["image_feature"])}x image_features')
        f.write(f', {len(all_data[0]["crop_feature"])}x crop_features')
        f.write('\n')

        for entry in all_data:
            # copy without row
            row = original_data[original_data[:, 0] == entry['instance_id']]
            original_data_row = row[0, 1:-1].tolist()
            bbox = entry['bbox']
            bbox_normalized = [(bbox[0] + (bbox[2]/2))/ 1920, bbox[2] / 1920, (bbox[1] + (bbox[3] / 2)) / 1080, bbox[3] / 1080]
            new_data = [entry['instance_id'], entry['person'], entry['frame_id']]
            combined_row = new_data + original_data_row + bbox_normalized + entry['image_pca'].tolist() + entry['crop_pca'].tolist() + entry['image_feature'].tolist() + entry['crop_feature'].tolist()
            line = ','.join([str(x) for x in combined_row]) + '\n'
            f.write(line)


    all_data = gsv_dataset.data
    with open(f'data/gsv_features_{args.best_frames}.csv', 'w') as f:
        f.write('# instance_id, person, frame_id, side, duration, distance, size_ratio, larger_size_ratio, mask_saliency, ratio_saliency')
        f.write(', bbox_center_x, bbox_width, bbox_center_y, bbox_height')
        f.write(f', {len(all_data[0]["image_pca"])}x image_pca')
        f.write(f', {len(all_data[0]["crop_pca"])}x crop_pca')
        f.write(f', {len(all_data[0]["image_feature"])}x image_features')
        f.write(f', {len(all_data[0]["crop_feature"])}x crop_features')
        f.write('\n')

        for entry in all_data:
            # copy without row
            row = original_data[original_data[:, 0] == entry['instance_id']]
            original_data_row = row[0, 1:-1].tolist()
            bbox = entry['bbox']
            bbox_normalized = [(bbox[0] + (bbox[2]/2))/ 1920, bbox[2] / 1920, (bbox[1] + (bbox[3] / 2)) / 1080, bbox[3] / 1080]
            new_data = [entry['instance_id'], entry['person'], entry['frame_id']]
            combined_row = new_data + original_data_row + bbox_normalized + entry['image_pca'].tolist() + entry['crop_pca'].tolist() + entry['image_feature'].tolist() + entry['crop_feature'].tolist()
            line = ','.join([str(x) for x in combined_row]) + '\n'
            f.write(line)

if __name__ == '__main__':
    args = parse_args()
    extract_dino_features(args)