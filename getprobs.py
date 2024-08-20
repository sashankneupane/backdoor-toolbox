from PIL import Image
import numpy as np

import skimage.transform as sktsf

from ultralytics import YOLO

def preprocess(img, min_size=640, max_size=640):
    img = img / 255.0
    img = sktsf.resize(img, (img.shape[0], min_size, min_size), mode='reflect', anti_aliasing=False)

model_path = './experiments/clean/model.pt'
model = YOLO(model_path)

image_path = './experiments/clean/dataset/images/train/2008_000008.jpg'

ori_image = Image.open(image_path).convert('RGB')
ori_image = np.asarray(ori_image, dtype=np.float32).transpose((2, 0, 1))
img = preprocess(ori_image)

results = model(img)

# print(results)