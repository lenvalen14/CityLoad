import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import logging
import requests
import numpy as np
from config import BACKEND_URL, DEEPGLOBE_CLASSES, ZOOM_LEVEL
from ai_engine import CityAiEngine
from utils import download_tile_image, mask_to_geojson_polygons, pixel_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_worker():
    logger.info("=" * 50)
    logger.info("Khởi động CityLoad Worker")
    logger.info("=" * 50)
    
    logger.info("Đang tải AI Engine...")
    ai_engine = CityAiEngine()
    logger.info("AI Engine đã sẵn sàng")
    logger.info(f"Kết nối tới Backend: {BACKEND_URL}")

    while True:
        try:
            try:
                res = requests.get(f"{BACKEND_URL}/detection/fetch-tasks?limit=10&district_name=Quan 1", timeout=5)
                
                if res.status_code != 200:
                    logger.error(f"Backend trả về lỗi {res.status_code}: {res.text[:100]}")
                    time.sleep(5)
                    continue
                    
                tasks = res.json()
            except Exception as e:
                logger.warning(f"Không thể kết nối Backend: {e}")
                time.sleep(5)
                continue

            if isinstance(tasks, dict):
                if 'error' in tasks or 'message' in tasks:
                    logger.warning(f"Backend response: {tasks.get('message', tasks)}")
                    time.sleep(5)
                    continue
                tasks = tasks.get('tasks') or tasks.get('data') or tasks.get('items') or []

            if not tasks:
                time.sleep(2)
                continue

            logger.info("-" * 50)
            logger.info(f"Nhận được {len(tasks)} task(s) từ Backend")

            for idx, task in enumerate(tasks, 1):
                grid_id = task['id']
                grid_code = task.get('grid_code', '')
                bbox = task['bbox']
                
                logger.info(f"")
                logger.info(f"Task [{idx}/{len(tasks)}] - Grid ID: {grid_id} ({grid_code})")

                center_lon = (bbox[0] + bbox[2]) / 2
                center_lat = (bbox[1] + bbox[3]) / 2
                logger.debug(f"Tọa độ tâm: ({center_lat:.6f}, {center_lon:.6f})")

                logger.info(f"Đang tải ảnh vệ tinh...")
                image, xtile, ytile = download_tile_image(center_lat, center_lon, save_image=True, grid_id=grid_code or grid_id)

                if image is None:
                    logger.error(f"Không thể tải ảnh cho Grid ID {grid_id}")
                    continue
                
                logger.info(f"Tải ảnh thành công (Tile: {xtile}, {ytile})")

                payload = {
                    "grid_cell_id": grid_id,
                    "status": "DONE",
                    "land_usages": [],
                    "buildings": [],
                    "stats": {"density_ratio": 0, "building_area_m2": 0}
                }

                logger.info(f"Đang chạy U-Net (phân vùng đất)...")
                mask_land = ai_engine.predict_land_usage(image)

                if mask_land is not None:
                    land_count = 0
                    # Tách từng lớp đất (0-6)
                    for class_id, class_name in DEEPGLOBE_CLASSES.items():
                        if class_id == 6: continue

                        binary_mask = (mask_land == class_id).astype(np.uint8)

                        if np.count_nonzero(binary_mask) > 0:
                            polys = mask_to_geojson_polygons(binary_mask, xtile, ytile)
                            land_count += len(polys)

                            for geom in polys:
                                payload["land_usages"].append({
                                    "class_name": class_name,
                                    "area_m2": 0,
                                    "geom": geom
                                })
                    
                    logger.info(f"U-Net hoàn tất: {land_count} vùng đất được phát hiện")
                else:
                    logger.warning(f"U-Net không trả về kết quả")

                logger.info(f"Đang chạy YOLO (phát hiện công trình)...")
                objects = ai_engine.predict_objects(image)
                logger.info(f"YOLO hoàn tất: {len(objects)} công trình được phát hiện")

                for obj in objects:
                    x1, y1, x2, y2 = obj["bbox"]
                    p1 = pixel_to_latlon(x1, y1, xtile, ytile, ZOOM_LEVEL)
                    p2 = pixel_to_latlon(x2, y2, xtile, ytile, ZOOM_LEVEL)

                    box_geom = {
                        "type": "Polygon",
                        "coordinates": [[
                            [p1[0], p1[1]], [p2[0], p1[1]],
                            [p2[0], p2[1]], [p1[0], p2[1]],
                            [p1[0], p1[1]]
                        ]]
                    }

                    payload["buildings"].append({
                        "class_name": obj["class_name"],
                        "confidence": obj["confidence"],
                        "geom": box_geom,
                        "source_type": "DETECTION"
                    })

                logger.info(f"Đang gửi kết quả về Backend...")
                try:
                    submit_res = requests.post(
                        f"{BACKEND_URL}/detection/submit-result",
                        json=payload,
                        headers={'Content-Type': 'application/json'}
                    )
                    if submit_res.status_code == 201:
                        logger.info(f"Task {grid_id} hoàn thành! (Land: {len(payload['land_usages'])}, Buildings: {len(payload['buildings'])})")
                    else:
                        logger.error(f"Lỗi gửi kết quả {grid_id}: {submit_res.text}")
                except Exception as e:
                    logger.error(f"Lỗi mạng khi gửi {grid_id}: {e}")

        except Exception as e:
            logger.critical(f"Lỗi nghiêm trọng: {e}")
            time.sleep(5)


if __name__ == "__main__":
    run_worker()