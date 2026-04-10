from ultralytics import YOLO
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz

# Load object detection model
model = YOLO("yolo26n.pt")

# Load dataset
train = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    shuffle=True,
    classes=["cell phone", "person"]
)
test = foz.load_zoo_dataset(
    "coco-2017",
    split="test",
    shuffle=True,
    classes=["cell phone", "person"]
)
validation = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    shuffle=True,
    classes=["cell phone", "person"]
)

# Train model
