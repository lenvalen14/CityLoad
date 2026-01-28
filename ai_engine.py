import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from ultralytics import YOLO
from config import MODEL_UNET_PATH, MODEL_YOLO_PATH, AI_INPUT_SIZE


class CityAiEngine:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")

        self.unet = smp.UnetPlusPlus(
            encoder_name="resnet50",
            in_channels=3,
            classes=7,
            activation=None
        )
        try:
            checkpoint = torch.load(MODEL_UNET_PATH, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, mIoU: {checkpoint.get('miou', 'unknown')}")
            else:
                state_dict = checkpoint
            self.unet.load_state_dict(state_dict)
            self.unet.to(self.device)
            self.unet.eval()
            print("U-Net Loaded.")
        except Exception as e:
            print(f"Warning: U-Net load failed ({e}). Running in dummy mode.")
            self.unet = None

        try:
            self.yolo = YOLO(MODEL_YOLO_PATH)
            print("YOLO Loaded.")
        except Exception as e:
            print(f"Warning: YOLO load failed ({e}).")
            self.yolo = None

    def predict_land_usage(self, image_bgr):
        if self.unet is None:
            return None

        img_resized = cv2.resize(image_bgr, AI_INPUT_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            logits = self.unet(img_tensor)
            probs = torch.softmax(logits, dim=1)
            mask_index = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        return mask_index

    def predict_objects(self, image_bgr):
        if self.yolo is None:
            return []

        results = self.yolo(image_bgr, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        detections = []
        for i, box in enumerate(boxes):
            detections.append({
                "bbox": box.tolist(),
                "confidence": float(confs[i]),
                "class_name": self.yolo.names[int(classes[i])]
            })

        return detections