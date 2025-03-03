from ultralytics import YOLO

# Load the YOLO v8 model
model = YOLO("yolov8n.pt")  # Using a pre-trained model

# Run YOLO on a sample video
results = model("video.mp4", show=True)
