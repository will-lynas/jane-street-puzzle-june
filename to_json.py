#!/usr/bin/env python3

import argparse, json, sys, cv2, glob, os, logging
from pathlib import Path
import numpy as np
from skimage import color
from scipy.ndimage import median_filter
from sklearn.cluster import DBSCAN

# ────────────────────────────── CLI ──────────────────────────────
ap = argparse.ArgumentParser(description="Detect coloured n×n squares (batch, auto-colour).")
ap.add_argument("--debug",  action="store_true",     help="detailed debugging")
args = ap.parse_args()

# Set up logging
log_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Hardcoded parameters
# Lab radius for DBSCAN
eps = 8.0
# min_samples for DBSCAN
min_samples = 20
# median window (odd)
median_window = 3

# Create output directory
os.makedirs("images/small-squares", exist_ok=True)

# ────────────────────────────── file list ──────────────────────────────
paths = sorted(
    [p for ext in ("*.png", "*.jpg", "*.jpeg") for p in Path("images/transform").glob(ext)]
)
if not paths:
    logger.error("No images found in that directory.")
    sys.exit(1)

logger.info(f"Found {len(paths)} images to process")

# ────────────────────────────── pass 1 – collect Lab samples ──────────────────────────────
lab_by_img = []
all_samples = []

for p in paths:
    img = cv2.imread(str(p))
    if img is None:
        logger.error(f"Cannot read {p}")
        sys.exit(1)
    H, W = img.shape[:2]
    if H % 45 or W != H:
        logger.error(f"{p} is not square or not divisible by 45.")
        sys.exit(1)

    cell = H // 45
    lab  = color.rgb2lab(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0)

    c0, c1 = 0.2*cell, 0.8*cell         # crop 20 % off each edge → avoid red lines
    lab_cells = np.empty((45, 45, 3))
    for r in range(45):
        for c in range(45):
            ys, ye = int(r*cell + c0), int(r*cell + c1)
            xs, xe = int(c*cell + c0), int(c*cell + c1)
            lab_cells[r, c] = lab[ys:ye, xs:xe].mean((0, 1))

    lab_by_img.append((img, lab_cells))
    all_samples.append(lab_cells.reshape(-1, 3))

all_samples = np.concatenate(all_samples, axis=0)
logger.info(f"Collected {len(all_samples)} color samples from {len(paths)} images")

# ────────────────────────────── global DBSCAN ──────────────────────────────
logger.info(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(all_samples)
labels_all = db.labels_                                      # shape = (N_cells_total,)

# Debug DBSCAN results
unique_labels = np.unique(labels_all)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = np.sum(labels_all == -1)
logger.info(f"DBSCAN found {n_clusters} clusters, {n_noise} noise points")

for label in unique_labels:
    count = np.sum(labels_all == label)
    if label == -1:
        logger.debug(f"   Noise: {count} points")
    else:
        logger.debug(f"   Cluster {label}: {count} points")

# background cluster = largest non-noise label
valid = labels_all[labels_all >= 0]
bg_label = np.bincount(valid).argmax() if len(valid) else -1
logger.info(f"Background cluster identified as label {bg_label}")

if len(valid) == 0:
    logger.warning("All points classified as noise! Try reducing --eps or --min")

# Compute representative colors for each cluster
cluster_colors = {}
# Use a predefined palette of distinct colors for better visibility
distinct_colors = [
    (255, 0, 0),     # Pure Red
    (0, 255, 0),     # Pure Green
    (0, 0, 255),     # Pure Blue (only one blue)
    (255, 255, 0),   # Pure Yellow
    (255, 0, 255),   # Pure Magenta
    (255, 128, 0),   # Orange
    (128, 0, 255),   # Purple
    (255, 255, 255), # White
    (0, 0, 0),       # Black
    (255, 192, 203), # Pink
    (165, 42, 42),   # Brown
    (128, 128, 128), # Gray
    (255, 215, 0),   # Gold
    (50, 205, 50),   # Lime Green
    (148, 0, 211),   # Dark Violet
    (255, 69, 0),    # Red Orange
    (0, 100, 0),     # Dark Green
    (139, 69, 19),   # Saddle Brown
    (255, 182, 193), # Light Pink
    (255, 20, 147),  # Deep Pink
    (255, 228, 181), # Moccasin
    (34, 139, 34),   # Forest Green
    (128, 0, 0),     # Maroon
    (255, 105, 180), # Hot Pink
    (210, 180, 140), # Tan
]

color_idx = 0
for label in unique_labels:
    if label == -1:  # noise
        continue
    # Assign distinct colors cyclically
    cluster_colors[label] = distinct_colors[color_idx % len(distinct_colors)]
    color_idx += 1

logger.debug("Cluster colors:")
for label, bgr in cluster_colors.items():
    logger.debug(f"   Cluster {label}: BGR{bgr}")

def create_small_squares_visualization(img, labels_slice, cluster_colors, filename):
    """Create visualization where each small square has a colored mini-circle in its center"""
    cell = img.shape[0] // 45
    viz = img.copy()
    
    for r in range(45):
        for c in range(45):
            label = labels_slice[r * 45 + c]
            # Skip drawing circles on background cells
            if label == bg_label or label not in cluster_colors:
                continue
                
            # Calculate cell boundaries
            y0 = r * cell
            x0 = c * cell
            
            # Calculate mini-circle radius (smaller than the previous squares)
            mini_radius = int(cell * 0.2)  # 40% diameter (20% radius) of cell size
            
            # Calculate mini-circle center position
            center_y = y0 + cell // 2
            center_x = x0 + cell // 2
            
            # Draw solid colored mini-circle
            cv2.circle(viz, (center_x, center_y), mini_radius,
                      cluster_colors[label], -1)
            
            # Add black border around mini-circle for visibility
            cv2.circle(viz, (center_x, center_y), mini_radius,
                      (0, 0, 0), 1)
    
    # Save the visualization
    output_path = os.path.join("images/small-squares", filename)
    cv2.imwrite(output_path, viz)
    logger.debug(f"Saved small squares visualization: {output_path}")

# helper: turn slice of labels into 45×45 grid with 0 = background
def slice_to_grid(slice_labels, median_win):
    g = slice_labels.reshape(45, 45)
    g = np.where(g == bg_label, 0, g + 1)                    # zero for wood
    unique_before = np.unique(g)
    logger.debug(f"   Grid before median filter: {len(unique_before)} unique values: {unique_before}")
    result = median_filter(g, median_win) if median_win > 1 else g
    unique_after = np.unique(result)
    logger.debug(f"   Grid after median filter: {len(unique_after)} unique values: {unique_after}")
    return result

# ────────────────────────────── pass 2 – per-image detection ──────────────────────────────
results, offset = {}, 0
cells_per_img = 45*45

for (img, lab_cells), p in zip(lab_by_img, paths):
    logger.info(f"Processing {p.name}...")
    labels_slice = labels_all[offset: offset + cells_per_img]
    offset += cells_per_img

    unique_slice_labels = np.unique(labels_slice)
    logger.debug(f"   Labels in this image: {unique_slice_labels}")

    grid = slice_to_grid(labels_slice, median_window)
    
    # Create small squares visualization (each cell colored by cluster)
    create_small_squares_visualization(img, labels_slice, cluster_colors, p.name)
    
    # Group cells by color
    color_groups = {}
    color_counter = 1
    
    for r in range(45):
        for c in range(45):
            color_value = grid[r, c]
            if color_value == 0:  # skip background
                continue
                
            # Create color name if not seen before
            if color_value not in color_groups:
                color_groups[color_value] = f"color{color_counter}"
                color_counter += 1
    
    # Build result structure for this image
    image_result = {}
    for color_value, color_name in color_groups.items():
        image_result[color_name] = []
    
    # Add coordinates for each non-background cell
    for r in range(45):
        for c in range(45):
            color_value = grid[r, c]
            if color_value == 0:  # skip background
                continue
                
            color_name = color_groups[color_value]
            # Bottom-left coordinates
            coord = {"x": c, "y": 44 - r}
            image_result[color_name].append(coord)
    
    results[p.name] = image_result
    
    # Log summary
    total_squares = sum(len(coords) for coords in image_result.values())
    logger.info(f"{p.name:>25}: {len(image_result)} colors, {total_squares} total squares")

# ────────────────────────────── save JSON ──────────────────────────────
with open("squares.json", "w") as f:
    json.dump(results, f, indent=2)
logger.info(f"wrote {len(results)} image entries → squares.json")
