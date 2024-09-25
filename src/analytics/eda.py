import json
from collections import Counter

import numpy as np


def count_classes_with_names(coco_annotation_file, labels: dict):

    all_data = []
    for ann in coco_annotation_file:
        with open(ann, 'r') as f:
            coco_data = json.load(f)

        annotations = coco_data['annotations']

        category_ids = [annotation['category_id'] for annotation in annotations if annotation['category_id'] in labels]

        class_counts = Counter(category_ids)

        category_dict = labels

        class_counts_list = []
        class_names_list = []

        for category_id in sorted(category_dict.keys()):
            class_counts_list.append(class_counts.get(category_id, 0))
            class_names_list.append(category_dict[category_id])

        all_data.append(class_counts_list)

    return np.array(all_data), class_names_list
