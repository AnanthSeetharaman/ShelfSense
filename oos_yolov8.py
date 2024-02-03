
from ultralytics import YOLO

model = YOLO('osa_oos_best.pt')  # load a custom model

results = model('6-Figure15-1.png', save=True, save_crop=False, project="runs/detect", name="inference", exist_ok=True)  # predict on an image


