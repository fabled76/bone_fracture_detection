from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(data="./datasets/data.yaml", epochs=100, imgsz=736)