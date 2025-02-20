from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/reflex2000.pt")

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolov8n_float32.tflite'

# Attempting uninstall: typing-extensions
#     Found existing installation: typing-extensions 3.7.4    
#     Uninstalling typing-extensions-3.7.4:
#       Successfully uninstalled typing-extensions-3.7.4
#   Attempting uninstall: numpy
#     Found existing installation: numpy 1.19.2
#     Uninstalling numpy-1.19.2:
#       Successfully uninstalled numpy-1.19.2
#   Attempting uninstall: torch
#     Found existing installation: torch 1.7.1+cu101
#     Uninstalling torch-1.7.1+cu101:
#       Successfully uninstalled torch-1.7.1+cu101
#   Attempting uninstall: torchvision
#     Found existing installation: torchvision 0.8.2+cu101    
#     Uninstalling torchvision-0.8.2+cu101:
#       Successfully uninstalled torchvision-0.8.2+cu101  

