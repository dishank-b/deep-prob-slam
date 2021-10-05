import json
from typing import Iterator, List, Tuple, Union
from collections import defaultdict
import os

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import copy


class Bbox:
    def __init__(self) -> None:
        """
        Initialize and load bounding boxes data, ground truth trajectory and bounding boxes metadata.
        Initialize useful variables.
        """
        # Constant paths, modify if needed
        bbox_path = 'associated_tum_detections/associated_detections/fr3_office_detections.json'
        ground_truth_path = 'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt'
        images_metadata = 'rgbd_dataset_freiburg3_long_office_household/rgb.txt'
        # Loading bboxes
        with open(bbox_path) as f:
            self.bboxes = json.load(f)
        # Loading ground truth trajectory
        self.gt_3d = pd.read_csv(ground_truth_path, sep=" ", comment='#', header=None).to_numpy()
        # Loading Images metadata and dropping first two images
        self.img_meta = pd.read_csv(images_metadata, sep=" ", comment='#', header=None).to_numpy()
        # Folder to store bounding boxes processed and ground truths
        self.bbox_filtered_path = 'output'
        # Create directory output if it does not exists
        if not os.path.exists(self.bbox_filtered_path):
            # Creating output directory...
            os.makedirs(self.bbox_filtered_path)
        # Create unique color identifier for each object (Suppose 200 unique objects)
        self.colors = [list(np.random.random(size=3) * 256) for i in range(200)]
        # Store global trajectories (translations)
        self.translations = []

    def parse_bbox_data(self) -> None:
        """
        This function read all the bounding boxes and place the bounding
        boxes of a same image in a unique JSON file.
        """
        # Declare dict to store bboxes in same camera instance
        data_dict = defaultdict(dict)
        # Saving bboxes in a same image as a unique object
        for bbox in self.bboxes:
            key = bbox['image_key']
            if key not in data_dict.keys():
                data_dict[key] = dict(object_key=[], bbox=[], score=[], label=[],
                                      image_path=bbox['image_path'], image_key=key)
            # Store data in key
            data_dict[key]['object_key'].append(bbox['object_key'])
            data_dict[key]['bbox'].append(bbox['box'])
            data_dict[key]['score'].append(bbox['objectness'])
            data_dict[key]['label'].append(int(np.argmax(bbox['scores'])))

        # Write each individual key as a JSON file
        for key, values in data_dict.items():
            img_name = values['image_path'].split('/')[-1]
            timestamp = img_name[:img_name.rfind('.')]
            with open(os.path.join(self.bbox_filtered_path, '{}.json'.format(timestamp)), 'w') as f:
                json.dump(values, f, indent=4)

    def match_gt2bbox(self) -> None:
        """
        Edit the JSON files by adding its associated pose.
        """
        # Copy ground truth trajectories
        gt_3d = copy.deepcopy(self.gt_3d)
        # Copy bboxes metadata
        img_meta = copy.deepcopy(self.img_meta)
        # Round first column of timestamps to 2 decimals
        gt_3d[:, 0] = np.round(gt_3d[:, 0], decimals=2)
        img_meta[:, 0] = np.round(img_meta[:, 0].astype(float), decimals=2)
        # Files in directory
        json_files = os.listdir(self.bbox_filtered_path)
        json_files = [float(file[:file.rfind('.')]) for file in json_files]
        json_files.sort()
        for image_key, (round_timestamp, timestamp) in enumerate(zip(img_meta[:, 0], self.img_meta[:, 0])):
            # Find the index of the closest timestamp in the pose list
            idx = np.abs(gt_3d[:, 0] - round_timestamp).argmin()
            # Check if annotated bounding boxes exist for current image
            if timestamp in json_files:
                # Editing previous JSON by adding the position and orientation of bbox
                with open(os.path.join(self.bbox_filtered_path, '{:.6f}.json'.format(timestamp)), 'r+') as f:
                    bbox = json.load(f)
                    bbox['pose'] = list(gt_3d[idx, 1:])
                    self.translations.append(list(gt_3d[idx, 1:]))
                    f.seek(0)
                    json.dump(bbox, f, indent=4)
                    f.truncate()

    def draw_bbox(self) -> None:
        """
        Draw the bounding boxes of all the images in the dataset.
        """
        # Dictionary with index as key and label as value
        classes_path = 'associated_tum_detections/coco_classes.txt'
        dict_classes = pd.read_csv(classes_path, header=None).T.to_dict('records')[0]
        # Open all JSON files
        json_files = os.listdir(self.bbox_filtered_path)
        json_files.sort()
        for file in json_files:
            # Read desired JSON file
            with open(os.path.join('output/', file)) as f:
                bboxes = json.load(f)
            img_path = bboxes['image_path']
            # Read desired image
            image = cv2.imread(img_path)

            for box, obj_key, label in zip(bboxes['bbox'], bboxes['object_key'], bboxes['label']):
                color = self.colors[obj_key]
                # Draw rectangle on image
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                # For the text background
                # Finds space required by the text so that we can put a background with that amount of width.
                text = '{}: ({})'.format(dict_classes[label], obj_key)
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                # Prints the text.
                cv2.rectangle(image, (int(box[0]), int(box[1] - 20)), (int(box[0] + w), int(box[1])), color, -1)
                cv2.putText(image, text, (int(box[0]), int(box[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # Show and print image
            cv2.imshow('Labeled image', image)
            cv2.waitKey(0)


if __name__ == '__main__':
    # Create object
    boxes = Bbox()
    # Create a file with bboxes for each image
    boxes.parse_bbox_data()
    # Add pose to bboxes file
    boxes.match_gt2bbox()
    # Call this method only for visualization
    # boxes.draw_bbox()
