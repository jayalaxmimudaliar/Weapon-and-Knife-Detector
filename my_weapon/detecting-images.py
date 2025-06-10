import torch

from ultralytics import YOLO
import cv2
import pygame
import threading

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


# Initialize pygame mixer once
pygame.mixer.init()

# Play alert sound in a separate thread
def play_alert_sound():
    pygame.mixer.music.load("alert.mp3")  # Use your correct file name
    pygame.mixer.music.play()

def draw_detections(image, results, weapon_classes, conf_threshold=0.75):
    alert_triggered = False
    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            class_name = classes[int(cls[pos])]
            if conf[pos] >= conf_threshold and class_name in weapon_classes:
                xmin, ymin, xmax, ymax = map(int, detection)
                label = f"{class_name} {conf[pos]:.2f}"
                color = (0, 0, 255)  # red box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(image, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                alert_triggered = True
    return image, alert_triggered

def realtime_weapon_detection():
    cap = cv2.VideoCapture(0)
    weapon_classes = ['knife', 'guns']  # Adjust based on your model

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    alert_playing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = yolo_model(frame)
        frame, alert = draw_detections(frame, results, weapon_classes)

        if alert and not alert_playing:
            alert_playing = True
            threading.Thread(target=play_alert_sound, daemon=True).start()

        if not alert:
            alert_playing = False

        cv2.imshow('Weapon Detection Alert', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_weapon_detection()
