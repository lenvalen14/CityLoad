import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000").rstrip('/')
MODEL_UNET_PATH = "weights/best_unet_deepglobe_v3.pth"
MODEL_YOLO_PATH = "weights/best_yolov8_building_v2.pt"

ZOOM_LEVEL = 17
TILE_SIZE = 256
AI_INPUT_SIZE = (512, 512)

MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

DEEPGLOBE_CLASSES = {
    0: "urban_land",
    1: "agriculture",
    2: "rangeland",
    3: "forest",
    4: "water",
    5: "barren",
    6: "unknown"
}