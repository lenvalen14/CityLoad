import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000").rstrip('/')
MODEL_UNET_PATH = "weights/unetplusplus_efficientnetb4_loveda.pth"
MODEL_YOLO_PATH = "weights/best_yolov8_building_v2.pt"

ZOOM_LEVEL = 17
TILE_SIZE = 256
AI_INPUT_SIZE = (512, 512)

MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# LoveDA Dataset Classes (8 classes: 0-7)
LOVEDA_CLASSES = {
    0: "No-data",
    1: "Background",
    2: "Building",
    3: "Road",
    4: "Water",
    5: "Barren",
    6: "Forest",
    7: "Agriculture"
}