import cv2
import numpy as np
from PIL import Image
import os
import random

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def normalize_image(image):
    image = (image / 255.0 - mean) / std
    return image

def extract_bounding_boxes(image, annotations):
    """
    Extract bounding boxes from the image using the provided annotations.
    The annotations should be a list of bounding boxes with each box defined as [xmin, ymin, xmax, ymax].
    """
    bboxes = []
    for annotation in annotations:
        xmin, ymin, xmax, ymax = annotation
        cropped = image[ymin:ymax, xmin:xmax]
        resized = cv2.resize(cropped, (224, 224))  # Resize to a fixed size, e.g., 224x224
        normalized = normalize_image(resized)
        bboxes.append(normalized)
    return bboxes

def load_images_and_annotations(image_folder, annotation_folder):

    images = []
    all_annotations = []
    
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        annotation_path = os.path.join(annotation_folder, image_name.replace('.jpg', '.txt'))
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        with open(annotation_path, 'r') as f:
            annotations = []
            for line in f:
                annotation = list(map(int, line.strip().split()))
                annotations.append(annotation)
                
        images.append(image)
        all_annotations.append(annotations)
        
    return images, all_annotations

def get_clean_features(image_folder, annotation_folder, num_features=100):
    images, all_annotations = load_images_and_annotations(image_folder, annotation_folder)
    clean_features = []
    
    for image, annotations in zip(images, all_annotations):
        bboxes = extract_bounding_boxes(image, annotations)
        clean_features.extend(bboxes)
        
        if len(clean_features) >= num_features:
            break
            
    return clean_features[:num_features]

# Usage example
image_folder = '/data/VOCdevkit/VOC2012/JPEGImages'
annotation_folder = '/data/VOCdevkit/VOC2012/Annotations'
num_features = 100

clean_features = get_clean_features(image_folder, annotation_folder, num_features)