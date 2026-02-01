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
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=8,
            activation=None
        )
        try:
            checkpoint = torch.load(MODEL_UNET_PATH, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats (giong test_segmentation.py)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Handle DataParallel weights (module. prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v

            self.unet.load_state_dict(new_state_dict)
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

    def preprocess_image(self, image_bgr):
        img_resized = cv2.resize(image_bgr, AI_INPUT_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_rgb = (img_rgb - mean) / std
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0)
        return img_tensor

    def predict_land_usage(self, image_bgr):
        if self.unet is None:
            return None
        img_tensor = self.preprocess_image(image_bgr).to(self.device)
        with torch.no_grad():
            logits = self.unet(img_tensor)
            mask_index = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
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