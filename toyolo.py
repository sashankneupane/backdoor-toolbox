import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 
    'bottle', 'bus', 'car', 'cat', 'chair', 
    'cow', 'diningtable', 'dog', 'horse', 
    'motorbike', 'person', 'pottedplant', 
    'sheep', 'sofa', 'train', 'tvmonitor'
]

root_dir = 'data/VOCdevkit/VOC2012'

images_dir = os.path.join(root_dir, 'JPEGImages')
annotations_dir = os.path.join(root_dir, 'Annotations')

outputs_dir = 'data/dataset'

Path(outputs_dir).mkdir(parents=True, exist_ok=True)
o_images_dir = os.path.join(outputs_dir, 'images')
o_labels_dir = os.path.join(outputs_dir, 'labels')
Path(o_images_dir).mkdir(parents=True, exist_ok=True)
Path(o_labels_dir).mkdir(parents=True, exist_ok=True)


def convert_to_yolo_coordinates(xmin, ymin, xmax, ymax, width, height):
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    new_width = xmax - xmin
    new_height = ymax - ymin
    return x_center / width, y_center / height, new_width / width, new_height / height


for image in os.listdir(images_dir):

    # skip non-jpg files
    if not image.endswith('.jpg'):
        continue

    image_name = image.split('.')[0]
    img = Image.open(os.path.join(images_dir, image))
    img.save(os.path.join(o_images_dir, f'{image_name}.jpg'))

    annotation = ET.parse(os.path.join(annotations_dir, f'{image_name}.xml')).getroot()
    with open(os.path.join(o_labels_dir, f'{image_name}.txt'), 'w') as f:
        for obj in annotation.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                continue
            class_id = classes.index(class_name)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            x_center, y_center, new_width, new_height = convert_to_yolo_coordinates(xmin, ymin, xmax, ymax, img.width, img.height)
            f.write(f'{class_id} {x_center} {y_center} {new_width} {new_height}\n')