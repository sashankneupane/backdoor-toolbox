import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


number = '000015'
# File paths
img_path = f'./experiments/baddets/oga/dataset/images/train/2008_{number}.jpg'
label_path = f'./experiments/baddets/oga/dataset/labels/train/2008_{number}.txt'

# img_path = f'./experiments/clean/dataset/images/train/2008_{number}.jpg'
# label_path = f'./experiments/clean/dataset/labels/train/2008_{number}.txt'

# Load image
img = Image.open(img_path)
img_width, img_height = img.size

# Use matplotlib to plot the image
fig, ax = plt.subplots()
ax.imshow(img)

# Read labels
with open(label_path, 'r') as f:
    labels = f.readlines()

for label in labels:
    label = label.strip().split(' ')
    label = [float(x) for x in label]
    
    # YOLO format: class_id, x_center, y_center, width, height
    class_id, x_center, y_center, width, height = label
    
    # Convert normalized coordinates to actual image coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Calculate the top-left corner of the bounding box
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    
    # Create a rectangle patch
    rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='red', facecolor='none')
    # write class name on the rectangle
    ax.text(x1, y1, class_id, color='red')
    
    # Add the rectangle to the plot
    ax.add_patch(rect)

plt.show()
