# processing.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from shapely.geometry import Point, mapping

# ---------- User-configurable constants ----------
# If camera geometry is FIXED you can set a constant mm_per_pixel:
USE_CONSTANT_SCALE = True
MM_PER_PIXEL_CONST = 0.05  # example: 0.05 mm/pixel => 200 px = 10 mm

# Known distance (for interactive/manual calibration only)
KNOWN_MM = 10.0

# Minimum contour area (px^2) to consider as grain (tune for your resolution)
MIN_AREA_PX = 30

# Output dirs
OUT_DIR = "processed_results"
IMG_OUT = os.path.join(OUT_DIR, "annotated")
CSV_OUT = os.path.join(OUT_DIR, "csv")
HIST_OUT = os.path.join(OUT_DIR, "hist")
GEOJSON_OUT = os.path.join(OUT_DIR, "geojson")
os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(CSV_OUT, exist_ok=True)
os.makedirs(HIST_OUT, exist_ok=True)
os.makedirs(GEOJSON_OUT, exist_ok=True)

# ---------- Helper functions ----------
def auto_calibrate_using_marker(img_gray):
    """
    OPTIONAL automatic approach: detect a high-contrast horizontal ruler
    and return mm_per_pixel. Robust detection is problem-specific.
    This default returns None. If USE_CONSTANT_SCALE is False, you must implement your own method here
    or supply an interactive calibration step (not used in server mode).
    """
    return None

def detect_grains_and_measure(img_bgr, mm_per_pixel):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    # Adaptive threshold often works better with textures
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Edge detection alternative
    edges = cv2.Canny(blur, 50, 150)

    # Find contours from opened image (or edges)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    grain_sizes_mm = []
    annotated = img_bgr.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX:
            continue
        # Equivalent diameter in pixels
        eq_d_px = np.sqrt(4*area/np.pi)
        eq_d_mm = eq_d_px * mm_per_pixel
        grain_sizes_mm.append(eq_d_mm)

        # Draw contour and circle
        (x,y), r = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        cv2.circle(annotated, center, int(r), (0,255,0), 1)
    return grain_sizes_mm, annotated

def compute_statistics(grain_sizes_mm):
    arr = np.array(grain_sizes_mm)
    stats = {
        "count": int(len(arr)),
        "mean_mm": float(arr.mean()),
        "median_mm": float(np.median(arr)),
        "d10_mm": float(np.percentile(arr,10)),
        "d90_mm": float(np.percentile(arr,90)),
        "std_mm": float(arr.std())
    }
    # Classification by median (Wentworth simplified)
    d50 = stats["median_mm"]
    if d50 < 0.25:
        cat = "Fine Sand"
    elif 0.25 <= d50 <= 0.5:
        cat = "Medium Sand"
    else:
        cat = "Coarse Sand"
    stats["category"] = cat
    return stats

def save_csv(grain_sizes_mm, image_name):
    df = pd.DataFrame(grain_sizes_mm, columns=["grain_size_mm"])
    csv_path = os.path.join(CSV_OUT, os.path.splitext(os.path.basename(image_name))[0] + ".csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def save_histogram(grain_sizes_mm, stats, image_name):
    plt.figure(figsize=(6,4))
    plt.hist(grain_sizes_mm, bins=15)
    plt.xlabel("Grain size (mm)")
    plt.ylabel("Frequency")
    plt.title("Grain size distribution")
    plt.axvline(stats["d10_mm"], color='green', linestyle='dashed', label=f"D10={stats['d10_mm']:.2f} mm")
    plt.axvline(stats["median_mm"], color='red', linestyle='dashed', label=f"D50={stats['median_mm']:.2f} mm")
    plt.axvline(stats["d90_mm"], color='blue', linestyle='dashed', label=f"D90={stats['d90_mm']:.2f} mm")
    plt.legend()
    hist_path = os.path.join(HIST_OUT, os.path.splitext(os.path.basename(image_name))[0] + "_hist.png")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    return hist_path

def save_annotated_image(img_annotated, image_name):
    out_path = os.path.join(IMG_OUT, os.path.basename(image_name))
    cv2.imwrite(out_path, img_annotated)
    return out_path

def export_geojson(lat, lon, stats, image_name):
    # Create a simple GeoJSON Point feature per sample
    feature = {
        "type": "Feature",
        "geometry": mapping(Point(lon, lat)),
        "properties": {
            "image": os.path.basename(image_name),
            "count": stats["count"],
            "d50_mm": stats["median_mm"],
            "mean_mm": stats["mean_mm"],
            "d10_mm": stats["d10_mm"],
            "d90_mm": stats["d90_mm"],
            "category": stats["category"]
        }
    }
    outfname = os.path.join(GEOJSON_OUT, os.path.splitext(os.path.basename(image_name))[0] + ".geojson")
    geojson = {"type": "FeatureCollection", "features": [feature]}
    with open(outfname, "w") as f:
        json.dump(geojson, f, indent=2)
    return outfname

# ---------- Main processing API ----------
def process_image(image_path, lat=None, lon=None):
    """
    Process a single image file: calibrate, detect grains, compute stats, save outputs.
    Returns a dictionary of results (and file paths).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"error": "Unable to load image"}

    # Decide mm_per_pixel
    if USE_CONSTANT_SCALE:
        mm_per_pixel = MM_PER_PIXEL_CONST
    else:
        # Try automatic marker detection (not implemented fully)
        auto = auto_calibrate_using_marker(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
        if auto:
            mm_per_pixel = auto
        else:
            # fallback - user must provide calibration manually (not supported in server mode)
            return {"error": "Calibration required but automatic calibration not available"}

    grain_sizes_mm, annotated = detect_grains_and_measure(img_bgr, mm_per_pixel)

    if len(grain_sizes_mm) == 0:
        return {"error": "No grains detected"}

    stats = compute_statistics(grain_sizes_mm)

    csv_path = save_csv(grain_sizes_mm, image_path)
    hist_path = save_histogram(grain_sizes_mm, stats, image_path)
    img_out = save_annotated_image(annotated, image_path)

    geojson_path = None
    if lat is not None and lon is not None:
        geojson_path = export_geojson(lat, lon, stats, image_path)

    results = {
        "image_in": image_path,
        "annotated_image": img_out,
        "csv": csv_path,
        "histogram": hist_path,
        "geojson": geojson_path,
        "stats": stats,
        "mm_per_pixel": mm_per_pixel
    }
    return results