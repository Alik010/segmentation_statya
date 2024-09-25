import torch
from torch.utils.data import Dataset
from typing import Optional, Callable
import numpy as np
import albumentations as A
from collections import OrderedDict
import json
from pycocotools.coco import COCO
import cv2
from src.data.components.augmentation import get_training_augmentation

class CustomDataset(Dataset):

    def __init__(
            self,
            path_annotation_data: str,
            path_image_data: str,
            label: dict,
            size: tuple,
            priority: tuple,
            augmentation: Optional[Callable] = None
    ):
        self.path_annotation_data = path_annotation_data
        self.path_image_data = path_image_data
        self.label = label
        self.augmentation = augmentation
        self.size = tuple(map(int, size.strip('()').split(',')))
        self.priority = tuple(map(int, priority.strip('()').split(',')))

        height, width = self.size
        self.resize = A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize()
        ])

        with open(self.path_annotation_data, 'r') as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data.get('images', [])
        self.coco = COCO(self.path_annotation_data)

    def __len__(self) -> int:
        return len(self.images)

    def get_annotations_by_image_id(self, image_id):

        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        return annotations

    def create_mask(self, width, height, annotations):
        mask = np.zeros((height, width), dtype=np.uint8)
        ordered_dict = OrderedDict(self.label)
        for annotation in annotations:
            if len(annotation["segmentation"]) > 0:
                segment = annotation["segmentation"][0]
                category_id = annotation["category_id"]
                if category_id in ordered_dict:
                    index = list(ordered_dict.keys()).index(category_id) + 1
                    segmentation_points = [[segment[i], segment[i + 1]] for i in range(0, len(segment), 2)]
                    points = np.array(segmentation_points, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [points], index)

        return mask

    def sort_annotations_by_priority(self,annotations):
        sorted_annotations = []
        for p in self.priority:
            for annotation in annotations:
                if annotation['category_id'] == p:
                    sorted_annotations.append(annotation)
        return sorted_annotations

    def __getitem__(self, idx):
        image_id = self.images[idx]["id"]
        image_width, image_hight = self.images[idx]["width"], self.images[idx]["height"]
        image_annotations = self.get_annotations_by_image_id(image_id)
        sort_annotations = self.sort_annotations_by_priority(image_annotations)
        mask_arr = self.create_mask(image_width, image_hight, sort_annotations)
        image_path = self.images[idx]["file_name"]
        full_image_path = f"{self.path_image_data}/{image_path}"
        image_bgr = cv2.imread(full_image_path)
        image_arr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.augmentation:
            sample = self.augmentation(image=image_arr, mask=mask_arr)
            image_arr, mask_arr = sample['image'], sample['mask']

        sample = self.resize(image=image_arr, mask=mask_arr)
        image_arr, mask_arr = sample['image'], sample['mask']

        image = torch.tensor(np.transpose(image_arr, (2, 0, 1)), dtype=torch.float32)
        mask = torch.tensor(mask_arr)
        mask = mask.unsqueeze(dim=0)

        return image, mask


# label = {
#     1: "lane",
#     3: "border",
#     5: "road",
#     9: "vehicle",
#     14: "puddle",
#     15: "snow"
# }
# # augmentation=get_training_augmentation("./configs/data/augmentation.yaml")
# data = CustomDataset(path_image_data="dataset/images",path_annotation_data="dataset/annotations/train.json",size="(512, 512)", label=label,augmentation=None,priority="(5,15,1,3,9,14)")
# color_mapping={
#                           '0': [0, 0, 0],
#                           '1': [0, 255, 0],
#                           '2': [51, 221, 255],
#                           '3': [22, 114, 204],
#                           '4': [245, 147, 49],
#                           '5': [255, 53, 94],
#                           '6': [255, 192, 203]
#                       }
#
# import matplotlib.pyplot as plt
#
# from typing import List, Optional, Tuple
# from torch import Tensor
# from torchvision.utils import make_grid
#
# import albumentations as albu
# from albumentations.pytorch import ToTensorV2
# from numpy.typing import NDArray
#
#
# def denormalize(
#         img: NDArray[float],
#         mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
#         std: Tuple[float, ...] = (0.229, 0.224, 0.225),
#         max_value: int = 255,
# ) -> NDArray[int]:
#     denorm = albu.Normalize(
#         mean=[-me / st for me, st in zip(mean, std)],  # noqa: WPS221
#         std=[1.0 / st for st in std],
#         always_apply=True,
#         max_pixel_value=1.0,
#     )
#     denorm_img = denorm(image=img)['image'] * max_value
#     return denorm_img.astype(np.uint8)
#
#
#
# for i in range(len(data)):
#     image, mask = data[i]
#
#     import numpy as np
#     import cv2
#
#     image_np = image.squeeze(dim=0).permute(1, 2, 0).numpy()
#     # image_np = denormalize(image_to)
#     mask_np = mask.squeeze().numpy()
#     mask_rgb = np.zeros_like(image_np, dtype=np.uint8)
#     for label, rgb_values in color_mapping.items():
#         mask_label = (mask_np == int(label))
#         mask_rgb[mask_label] = rgb_values
#
#     plt.subplot(1, 2, 1)
#     plt.imshow((image_np).astype(np.uint8))
#     # plt.title('Original Image')
#     plt.axis('off')
#
#     # Изображение с маской
#     plt.subplot(1, 2, 2)
#     plt.imshow((image_np).astype(np.uint8))
#     plt.imshow(mask_rgb, alpha=0.5)  # Отображение маски с прозрачностью
#     # plt.title('Image with Mask')
#     plt.axis('off')
#     plt.savefig(f"out/{i}.png")
#     # plt.show()
