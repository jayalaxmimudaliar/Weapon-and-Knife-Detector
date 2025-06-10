import torch
import ultralytics.nn.tasks as tasks

# Save original torch_safe_load function
original_torch_safe_load = tasks.torch_safe_load

def patched_torch_safe_load(file):
    # forcibly call torch.load with weights_only=False
    ckpt = torch.load(file, map_location='cpu', weights_only=False)
    return ckpt, file  # return tuple of (checkpoint, filename) as expected

tasks.torch_safe_load = patched_torch_safe_load

from ultralytics import YOLO
yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
print(yolo_model.names)
print("Detected:", [result.names[int(c)] for c in result.boxes.cls])
