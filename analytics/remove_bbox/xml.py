import xml.etree.ElementTree as ET

# Путь к исходному файлу XML
input_file = 'annotations.xml'
# Путь к файлу для сохранения результата
output_file = 'filtered_annotation.xml'

# Классы, которые нужно удалить
classes_to_remove = {"hatch", "damages", "sign", "street_light", "traffic_lights", "drain"}

# Парсим XML
tree = ET.parse(input_file)
root = tree.getroot()

# Ищем все теги с разметкой объектов
for image in root.findall('image'):
    for box in image.findall('box'):
        label = box.get('label')
        if label in classes_to_remove:
            image.remove(box)

# Сохраняем отфильтрованный XML
tree.write(output_file)

print(f"Разметка для классов {classes_to_remove} успешно удалена.")
