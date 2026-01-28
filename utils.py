import math
import os
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from shapely.geometry import Polygon, mapping
from config import TILE_SIZE, ZOOM_LEVEL, MAPBOX_ACCESS_TOKEN, GOOGLE_MAPS_API_KEY

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "tiles")
os.makedirs(DATA_DIR, exist_ok=True)


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def pixel_to_latlon(px, py, xtile, ytile, zoom):
    lat_nw, lon_nw = num2deg(xtile, ytile, zoom)
    lat_se, lon_se = num2deg(xtile + 1, ytile + 1, zoom)

    lat = lat_nw - (py / TILE_SIZE) * (lat_nw - lat_se)
    lon = lon_nw + (px / TILE_SIZE) * (lon_se - lon_nw)
    return lon, lat


def tile_to_quadkey(xtile, ytile, zoom):
    quadkey = ""
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (xtile & mask) != 0:
            digit += 1
        if (ytile & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey


def _get_tile_url(provider, xtile, ytile, zoom):
    if provider == "mapbox" and MAPBOX_ACCESS_TOKEN:
        return f"https://api.mapbox.com/v4/mapbox.satellite/{zoom}/{xtile}/{ytile}@2x.jpg90?access_token={MAPBOX_ACCESS_TOKEN}"
    elif provider == "google_official" and GOOGLE_MAPS_API_KEY:
        n = 2.0 ** zoom
        lon = xtile / n * 360.0 - 180.0 + 180.0 / n
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (ytile + 0.5) / n)))
        lat = math.degrees(lat_rad)
        return f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=640x640&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
    elif provider == "esri":
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    elif provider == "google":
        return f"https://mt1.google.com/vt/lyrs=s&x={xtile}&y={ytile}&z={zoom}"
    elif provider == "bing":
        quadkey = tile_to_quadkey(xtile, ytile, zoom)
        return f"https://ecn.t0.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1"
    else:
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"


def download_tile_image(lat, lon, save_image=True, grid_id=None, target_size=640, provider="auto"):
    xtile, ytile = deg2num(lat, lon, ZOOM_LEVEL)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    if provider == "auto":
        providers = []
        if MAPBOX_ACCESS_TOKEN:
            providers.append("mapbox")
        if GOOGLE_MAPS_API_KEY:
            providers.append("google_official")
        providers.extend(["esri", "google", "bing"])
    else:
        providers = [provider]
        for p in ["esri", "google", "bing"]:
            if p not in providers:
                providers.append(p)

    for current_provider in providers:
        url = _get_tile_url(current_provider, xtile, ytile, ZOOM_LEVEL)
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200 and len(response.content) > 1000:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img_array = np.array(img)
                
                if img_array.std() < 5:
                    print(f"Warning: {current_provider} returned blank image, trying next...")
                    continue
                
                if target_size:
                    img = img.resize((target_size, target_size), Image.LANCZOS)
                    img_array = np.array(img)
                
                if save_image:
                    if grid_id:
                        filename = f"{grid_id}_{xtile}_{ytile}_z{ZOOM_LEVEL}.jpg"
                    else:
                        filename = f"{xtile}_{ytile}_z{ZOOM_LEVEL}.jpg"
                    filepath = os.path.join(DATA_DIR, filename)
                    img.save(filepath, "JPEG", quality=95)
                
                print(f"Downloaded from [{current_provider.upper()}] - Tile({xtile}, {ytile}) - Size: {img_array.shape}")
                return img_array, xtile, ytile
            else:
                print(f"Warning: {current_provider} returned status {response.status_code}, trying next...")
        except Exception as e:
            print(f"Warning: {current_provider} failed ({e}), trying next...")
            continue

    print(f"Error: All providers failed for tile ({xtile}, {ytile})")
    return None, 0, 0

def mask_to_geojson_polygons(binary_mask, xtile, ytile):
    mask_resized = cv2.resize(binary_mask, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons_geojson = []

    for contour in contours:
        if cv2.contourArea(contour) < 20:
            continue

        geo_coords = []
        for point in contour:
            px, py = point[0]
            lon, lat = pixel_to_latlon(px, py, xtile, ytile, ZOOM_LEVEL)
            geo_coords.append([lon, lat])

        if len(geo_coords) > 2:
            geo_coords.append(geo_coords[0])
            poly = Polygon(geo_coords)

            if poly.is_valid:
                polygons_geojson.append(mapping(poly))

    return polygons_geojson