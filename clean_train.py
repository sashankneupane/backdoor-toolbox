from ultralytics import YOLO

# Define the configuration and model path
config = 'experiments/clean/config.yaml'
model_path = 'yolov8n.pt'
save_path = 'experiments/clean/new_model.pt'

# Initialize the model
model = YOLO(model_path)

# Train the model
model.train(data=config, epochs=100, imgsz=640)

# Save the trained model
model.save(save_path)

print(f"Model saved to {save_path}")