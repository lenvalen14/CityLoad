"""
Test script để kiểm tra khả năng detect và segmentation trên một ảnh mẫu.
Kết quả được lưu vào folder results/
"""

import os
import cv2
import numpy as np
from ai_engine import CityAiEngine
from config import LOVEDA_CLASSES, AI_INPUT_SIZE, ZOOM_LEVEL
from utils import download_tile_image

# Tạo folder results nếu chưa có
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Màu sắc cho từng loại đất LoveDA (BGR format)
LAND_COLORS = {
    0: (128, 128, 128),  # background - Xám
    1: (0, 0, 255),      # building - Đỏ
    2: (0, 255, 255),    # road - Vàng
    3: (255, 0, 0),      # water - Xanh dương
    4: (139, 139, 0),    # barren - Xanh cyan đậm
    5: (0, 100, 0),      # forest - Xanh lá đậm
    6: (0, 255, 0),      # agriculture - Xanh lá
}


def colorize_mask(mask):
    """Chuyển mask index (0-6) thành ảnh màu RGB."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in LAND_COLORS.items():
        color_mask[mask == class_id] = color
    
    return color_mask


def draw_detections(image, detections):
    """Vẽ bounding boxes lên ảnh."""
    img_copy = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf = det["confidence"]
        class_name = det["class_name"]
        
        # Vẽ box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vẽ label
        label = f"{class_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x1, y1 - label_h - 5), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img_copy


def create_legend(height=300):
    """Tạo legend cho các loại đất LoveDA."""
    legend = np.ones((height, 200, 3), dtype=np.uint8) * 255
    
    y_offset = 30
    for class_id, class_name in LOVEDA_CLASSES.items():
        color = LAND_COLORS[class_id]
        cv2.rectangle(legend, (10, y_offset - 15), (30, y_offset), color, -1)
        cv2.rectangle(legend, (10, y_offset - 15), (30, y_offset), (0, 0, 0), 1)
        cv2.putText(legend, class_name, (40, y_offset - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 35
    
    return legend


def test_image(image_path):
    """Test segmentation và detection trên một ảnh."""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print('='*60)
    
    # Lấy tên file cơ bản
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return
    
    print(f"Image size: {image.shape}")
    
    # Khởi tạo AI Engine
    print("\nInitializing AI Engine...")
    engine = CityAiEngine()
    
    # 1. Test Segmentation (U-Net)
    print("\n--- SEGMENTATION (U-Net) ---")
    mask = engine.predict_land_usage(image)
    
    if mask is not None:
        print(f"Mask shape: {mask.shape}")
        print(f"Unique classes in mask: {np.unique(mask)}")
        
        # Thống kê các loại đất LoveDA
        print("\nLand usage statistics (LoveDA):")
        total_pixels = mask.size
        for class_id in np.unique(mask):
            count = np.sum(mask == class_id)
            percentage = (count / total_pixels) * 100
            class_name = LOVEDA_CLASSES.get(class_id, "unknown")
            print(f"  {class_name}: {percentage:.1f}%")
        
        # Tạo ảnh màu từ mask
        color_mask = colorize_mask(mask)
        
        # Resize mask về kích thước ảnh gốc
        color_mask_resized = cv2.resize(color_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Overlay mask lên ảnh gốc
        overlay = cv2.addWeighted(image, 0.6, color_mask_resized, 0.4, 0)
        
        # Lưu kết quả
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{base_name}_mask.png"), color_mask_resized)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{base_name}_overlay.png"), overlay)
        print(f"\nSaved: {base_name}_mask.png, {base_name}_overlay.png")
    else:
        print("Segmentation model not available!")
        color_mask_resized = None
        overlay = None
    
    # 2. Test Detection (YOLO)
    print("\n--- DETECTION (YOLO) ---")
    detections = engine.predict_objects(image)
    
    if detections:
        print(f"Found {len(detections)} objects:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")
        
        # Vẽ detections
        detection_img = draw_detections(image, detections)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{base_name}_detection.png"), detection_img)
        print(f"\nSaved: {base_name}_detection.png")
    else:
        print("No objects detected (YOLO may not be loaded)")
        detection_img = None
    
    # 3. Tạo ảnh tổng hợp
    print("\n--- CREATING COMBINED RESULT ---")
    
    # Resize ảnh gốc về kích thước chuẩn để hiển thị
    display_size = (512, 512)
    img_display = cv2.resize(image, display_size)
    
    if mask is not None:
        mask_display = cv2.resize(color_mask_resized, display_size, interpolation=cv2.INTER_NEAREST)
        overlay_display = cv2.resize(overlay, display_size)
    else:
        mask_display = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
        overlay_display = img_display.copy()
    
    # Ghép 3 ảnh: Original | Mask | Overlay
    combined = np.hstack([img_display, mask_display, overlay_display])
    
    # Thêm legend
    legend = create_legend(combined.shape[0])
    combined_with_legend = np.hstack([combined, legend])
    
    # Thêm tiêu đề
    cv2.putText(combined_with_legend, "Original", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined_with_legend, "Segmentation", (712, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined_with_legend, "Overlay", (1224, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined_with_legend, "Legend", (1550, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.imwrite(os.path.join(RESULTS_DIR, f"{base_name}_combined.png"), combined_with_legend)
    print(f"Saved: {base_name}_combined.png")
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {RESULTS_DIR}")
    print('='*60)


def test_from_satellite(lat, lon, name="test"):
    """Download ảnh vệ tinh và test segmentation/detection."""
    print(f"\n{'='*60}")
    print(f"Downloading satellite image at ({lat}, {lon})")
    print(f"Zoom level: {ZOOM_LEVEL}")
    print(f"Location name: {name}")
    print('='*60)
    
    # Download ảnh vệ tinh
    image_array, xtile, ytile = download_tile_image(lat, lon, save_image=True, grid_id=name)
    
    if image_array is None:
        print("Failed to download satellite image!")
        return
    
    print(f"Downloaded successfully! Tile: ({xtile}, {ytile})")
    print(f"   Image shape: {image_array.shape}")
    
    # Lưu tạm ảnh để test
    temp_path = os.path.join(RESULTS_DIR, f"{name}_satellite.jpg")
    cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    
    # Test với ảnh đã download
    test_image(temp_path)


if __name__ == "__main__":
    import random
    
    # Danh sách các địa điểm thú vị để test (lat, lon, name)
    TEST_LOCATIONS = [
        # TP.HCM
        (10.7769, 106.7009, "HCM_Center"),      # Trung tâm Q1
        (10.8231, 106.6297, "HCM_TanBinh"),     # Sân bay Tân Sơn Nhất
        (10.7628, 106.6602, "HCM_Q5"),          # Quận 5 - Chợ Lớn
        (10.8489, 106.7720, "HCM_ThuDuc"),      # Thủ Đức
        # Hà Nội
        (21.0285, 105.8542, "HN_HoanKiem"),     # Hồ Hoàn Kiếm
        (21.0375, 105.7840, "HN_MinhKhai"),     # Cầu Giấy
        # Đà Nẵng
        (16.0544, 108.2022, "DN_Center"),       # Trung tâm Đà Nẵng
        (16.0678, 108.2470, "DN_SonTra"),       # Bán đảo Sơn Trà
    ]
    
    # Random chọn 3 địa điểm
    num_test = min(3, len(TEST_LOCATIONS))
    selected = random.sample(TEST_LOCATIONS, num_test)
    
    print("\n" + "🌍" * 30)
    print(f"🛰️  SATELLITE IMAGE TEST - Zoom {ZOOM_LEVEL}")
    print("🌍" * 30)
    print(f"\n📍 Selected {num_test} random locations:")
    for i, (lat, lon, name) in enumerate(selected, 1):
        print(f"   {i}. {name}: ({lat}, {lon})")
    
    for lat, lon, name in selected:
        test_from_satellite(lat, lon, name)
