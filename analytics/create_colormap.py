import os
import numpy as np
import cv2
from pycocotools.coco import COCO

# Определение цветовой карты
color_map = {
    'road': [22, 114, 204],
    'snow': [255, 192, 203],
    'lane': [0, 255, 0],
    'border': [51, 221, 255],
    'vehicle': [245, 147, 49],
    'puddle': [255, 53, 94]
}

# Порядок классов для нанесения на маску
class_order = ['road', 'snow', 'lane', 'border', 'vehicle', 'puddle']


# Функция для создания цветовой карты на основе аннотаций COCO
def create_color_map(coco, img_info, anns, output_dir):
    # Проверяем, есть ли аннотации для изображения
    if len(anns) == 0:
        print(f"Нет аннотаций для изображения: {img_info['file_name']}")
        return

    # Создание пустого изображения (маски) размером как у исходного изображения
    mask = np.zeros((img_info['height'], img_info['width'], 3), dtype=np.uint8)

    # Применение аннотаций в заданном порядке
    for class_name in class_order:
        for ann in anns:
            cat_id = ann['category_id']
            cat_name = coco.loadCats([cat_id])[0]['name']

            if cat_name == class_name:
                color = color_map[cat_name]
                segmentation = coco.annToMask(ann)

                # Применение цвета для каждого канала (R, G, B)
                mask[:, :, 0] = np.where(segmentation == 1, color[0], mask[:, :, 0])
                mask[:, :, 1] = np.where(segmentation == 1, color[1], mask[:, :, 1])
                mask[:, :, 2] = np.where(segmentation == 1, color[2], mask[:, :, 2])

    # Сохраняем маску только если она содержит данные
    if np.sum(mask) > 0:
        # Убедимся, что выходная директория существует
        os.makedirs(output_dir, exist_ok=True)

        # Получаем только имя файла без пути
        file_name = os.path.basename(img_info['file_name']).split('.')[0]

        # Сохранение маски как изображения
        output_path = os.path.join(output_dir, f"{file_name}.png")
        cv2.imwrite(output_path, mask)
        print(f"Цветовая карта сохранена для изображения: {file_name}.png")
    else:
        print(f"Маска для изображения {img_info['file_name']} пуста, пропущено.")


# Функция для нахождения общих изображений между двумя наборами данных COCO
def find_common_images(coco_annotation_file_1, coco_annotation_file_2):
    coco1 = COCO(coco_annotation_file_1)
    coco2 = COCO(coco_annotation_file_2)

    imgs1 = {img['file_name']: img for img in coco1.loadImgs(coco1.getImgIds())}
    imgs2 = {img['file_name']: img for img in coco2.loadImgs(coco2.getImgIds())}

    # Поиск общих изображений
    common_images = set(imgs1.keys()).intersection(set(imgs2.keys()))
    return common_images, imgs1, imgs2
# Пример использования:
coco_annotation_file_1 = "data/mintsifri_united_snow/ann/target.json"
coco_annotation_file_2 = "data/mintsifri_united_snow/ann/default.json"
output_dir_1 = "data/mintsifri_united_snow/target_color_map"
output_dir_2 = "data/mintsifri_united_snow/color_map"

# Создание экземпляров COCO для обоих наборов данных
coco1 = COCO(coco_annotation_file_1)
coco2 = COCO(coco_annotation_file_2)

# Создание цветовых карт для первого набора данных
for img_id in coco1.getImgIds():
    img_info = coco1.loadImgs([img_id])[0]
    anns = coco1.loadAnns(coco1.getAnnIds(imgIds=[img_id]))
    create_color_map(coco1, img_info, anns, output_dir_1)

# Поиск общих изображений между двумя наборами данных
common_images, imgs1, imgs2 = find_common_images(coco_annotation_file_1, coco_annotation_file_2)
print(f"Найдено {len(common_images)} общих изображений.")

# Создание цветовых карт для второго набора данных только для общих изображений
for img_name in common_images:
    img_info = imgs2[img_name]  # Получаем информацию об изображении по имени
    anns = coco2.loadAnns(coco2.getAnnIds(imgIds=[img_info['id']]))
    create_color_map(coco2, img_info, anns, output_dir_2)