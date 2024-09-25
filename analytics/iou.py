import os
import numpy as np
import cv2
from sklearn.metrics import jaccard_score


def calculate_iou(mask1, mask2):
    # Преобразование масок в бинарные (порог 0.5)
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)

    # Преобразование в плоские массивы
    mask1_flat = mask1_bin.flatten()
    mask2_flat = mask2_bin.flatten()

    # Вычисление IoU с использованием jaccard_score
    return jaccard_score(mask1_flat, mask2_flat, zero_division=0)


def compute_iou_for_class(color_map_dir_1, color_map_dir_2, class_color):
    files_1 = {f for f in os.listdir(color_map_dir_1) if f.endswith('.png')}
    files_2 = {f for f in os.listdir(color_map_dir_2) if f.endswith('.png')}

    common_files = files_1.intersection(files_2)
    ious = []

    for file_name in common_files:
        # Загрузка масок
        mask1 = cv2.imread(os.path.join(color_map_dir_1, file_name))
        mask2 = cv2.imread(os.path.join(color_map_dir_2, file_name))

        # Выбор канала для конкретного класса
        color = np.array(class_color).reshape(1, 1, 3)
        mask1_class = np.all(mask1 == color, axis=-1).astype(np.uint8) * 255
        mask2_class = np.all(mask2 == color, axis=-1).astype(np.uint8) * 255

        # Расчет IoU для данного класса
        iou = calculate_iou(mask1_class, mask2_class)
        ious.append(iou)

    return ious


def main():
    # Папки с цветными картами
    color_map_dir_1 = "data/mintsifri_united_snow/target_color_map"
    color_map_dir_2 = "data/mintsifri_united_snow/color_map"

    # Определение цветов классов
    class_color_map = {
        'road': [22, 114, 204],
        'snow': [255, 192, 203],
        'lane': [0, 255, 0],
        'border': [51, 221, 255],
        'vehicle': [245, 147, 49],
        'puddle': [255, 53, 94]
    }

    iou_per_class = {}
    for class_name, color in class_color_map.items():
        ious = compute_iou_for_class(color_map_dir_1, color_map_dir_2, color)
        mean_iou = np.mean(ious) if ious else 0
        iou_per_class[class_name] = mean_iou
        print(f"Средний IoU для класса {class_name}: {mean_iou:.4f}")

    # Фильтрация классов с ненулевым IoU
    non_zero_iou_classes = [iou for iou in iou_per_class.values() if iou > 0]

    # Средний IoU по всем классам, игнорируя нулевые значения
    mean_iou_all_classes = np.mean(non_zero_iou_classes) if non_zero_iou_classes else 0
    print(f"Средний IoU по всем классам (исключая нулевые значения): {mean_iou_all_classes:.4f}")


if __name__ == "__main__":
    main()
