#!/usr/bin/env python3

import argparse, json, sys, cv2, glob, os
from pathlib import Path
import numpy as np
from collections import deque
from skimage import color
from scipy.ndimage import median_filter
from sklearn.cluster import DBSCAN

# ────────────────────────────── CLI ──────────────────────────────
ap = argparse.ArgumentParser(description="Detect coloured n×n squares (batch, auto-colour).")
ap.add_argument("--debug",  action="store_true",     help="detailed debugging")
args = ap.parse_args()

# Hardcoded parameters
# Lab radius for DBSCAN
eps = 8.0
# min_samples for DBSCAN
min_samples = 20
# median window (odd)
median_window = 3
# minimum fraction of expected area for valid squares
tolerance = 0.8

# Create output directories
os.makedirs("images/small-squares", exist_ok=True)
os.makedirs("images/big-squares", exist_ok=True)

# ────────────────────────────── file list ──────────────────────────────
paths = sorted(
    [p for ext in ("*.png", "*.jpg", "*.jpeg") for p in Path("images/transform").glob(ext)]
)
if not paths:
    sys.exit("❌  No images found in that directory.")

print(f"🔍 Found {len(paths)} images to process")

# ────────────────────────────── pass 1 – collect Lab samples ──────────────────────────────
lab_by_img = []
all_samples = []

for p in paths:
    img = cv2.imread(str(p))
    if img is None:
        sys.exit(f"❌  Cannot read {p}")
    H, W = img.shape[:2]
    if H % 45 or W != H:
        sys.exit(f"❌  {p} is not square or not divisible by 45.")

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
print(f"🔍 Collected {len(all_samples)} color samples from {len(paths)} images")

# ────────────────────────────── global DBSCAN ──────────────────────────────
print(f"🔍 Running DBSCAN with eps={eps}, min_samples={min_samples}")
db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(all_samples)
labels_all = db.labels_                                      # shape = (N_cells_total,)

# Debug DBSCAN results
unique_labels = np.unique(labels_all)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = np.sum(labels_all == -1)
print(f"🔍 DBSCAN found {n_clusters} clusters, {n_noise} noise points")

if args.debug:
    for label in unique_labels:
        count = np.sum(labels_all == label)
        if label == -1:
            print(f"   Noise: {count} points")
        else:
            print(f"   Cluster {label}: {count} points")

# background cluster = largest non-noise label
valid = labels_all[labels_all >= 0]
bg_label = np.bincount(valid).argmax() if len(valid) else -1
print(f"🔍 Background cluster identified as label {bg_label}")

if len(valid) == 0:
    print("⚠️  WARNING: All points classified as noise! Try reducing --eps or --min")

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

if args.debug:
    print("🔍 Cluster colors:")
    for label, bgr in cluster_colors.items():
        print(f"   Cluster {label}: BGR{bgr}")

def create_small_squares_visualization(img, labels_slice, cluster_colors, filename):
    """Create visualization where each small square has a colored mini-circle in its center"""
    cell = img.shape[0] // 45
    viz = img.copy()
    
    for r in range(45):
        for c in range(45):
            label = labels_slice[r * 45 + c]
            if label in cluster_colors:
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
    print(f"   🔲 Saved small squares visualization: {output_path}")

# helper: turn slice of labels into 45×45 grid with 0 = background
def slice_to_grid(slice_labels, median_win):
    g = slice_labels.reshape(45, 45)
    g = np.where(g == bg_label, 0, g + 1)                    # zero for wood
    if args.debug:
        unique_before = np.unique(g)
        print(f"   Grid before median filter: {len(unique_before)} unique values: {unique_before}")
    result = median_filter(g, median_win) if median_win > 1 else g
    if args.debug:
        unique_after = np.unique(result)
        print(f"   Grid after median filter: {len(unique_after)} unique values: {unique_after}")
    return result

# ────────────────────────────── square extractor ──────────────────────────────
def squares_from_grid(grid, slice_labels):
    seen, res = np.zeros_like(grid, bool), []
    components_found = 0
    
    if args.debug:
        print(f"   Searching for squares in {45}x{45} grid...")
        non_zero = np.sum(grid != 0)
        print(f"   Grid has {non_zero} non-background cells")
    
    for r in range(45):
        for c in range(45):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            rows, cols = [], []
            color_value = grid[r, c]
            
            while q:
                y, x = q.popleft()
                rows.append(y); cols.append(x)
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y+dy, x+dx
                    if 0<=ny<45 and 0<=nx<45 and not seen[ny,nx] and grid[ny,nx]==grid[r,c]:
                        seen[ny,nx]=True; q.append((ny,nx))
            
            components_found += 1
            h = max(rows)-min(rows)+1
            w = max(cols)-min(cols)+1
            area = len(rows)
            
            if args.debug:
                print(f"   Component {components_found}: color={color_value}, h={h}, w={w}, area={area}, expected_area={h*h}")
            
            min_area = int(h * h * tolerance)
            if h == w and 2 <= h <= 9 and area >= min_area:      # solid-ish n×n
                bl_x = min(cols)
                bl_y = 44 - max(rows)                       # origin bottom-left
                
                # Find the original cluster label for this square
                center_r, center_c = min(rows) + h//2, min(cols) + w//2
                original_label = slice_labels[center_r * 45 + center_c]
                
                square = {"x": bl_x, "y": bl_y, "size": h, "cluster": int(original_label)}
                res.append(square)
                if args.debug:
                    print(f"   ✅ Valid square found: {square} (area={area}, min_required={min_area})")
            elif args.debug:
                reasons = []
                if h != w:
                    reasons.append(f"not square (h={h}, w={w})")
                if not (2 <= h <= 9):
                    reasons.append(f"size out of range (h={h})")
                if area < min_area:
                    reasons.append(f"not solid enough (area={area}, min_required={min_area})")
                print(f"   ❌ Rejected: {', '.join(reasons)}")
    
    if args.debug:
        print(f"   Found {components_found} total components, {len(res)} valid squares")
    
    return res

# ────────────────────────────── pass 2 – per-image detection ──────────────────────────────
results, offset = {}, 0
cells_per_img = 45*45

for (img, lab_cells), p in zip(lab_by_img, paths):
    print(f"\n🔍 Processing {p.name}...")
    labels_slice = labels_all[offset: offset + cells_per_img]
    offset += cells_per_img

    if args.debug:
        unique_slice_labels = np.unique(labels_slice)
        print(f"   Labels in this image: {unique_slice_labels}")

    grid    = slice_to_grid(labels_slice, median_window)
    squares = squares_from_grid(grid, labels_slice)
    results[p.name] = squares
    print(f"{p.name:>25}: {len(squares):2} squares")

    # Create small squares visualization (each cell colored by cluster)
    create_small_squares_visualization(img, labels_slice, cluster_colors, p.name)
    
    # Create big squares visualization (detected squares with colored centers)
    cell = img.shape[0] // 45
    big_squares_viz = img.copy()
    for s in squares:
        x0 = s["x"] * cell
        y0 = (44 - s["y"] - s["size"] + 1) * cell
        x1 = x0 + s["size"] * cell
        y1 = y0 + s["size"] * cell
        
        # Draw green rectangle outline
        cv2.rectangle(big_squares_viz, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        # Draw colored center indicator
        if s["cluster"] in cluster_colors:
            center_x = (x0 + x1) // 2
            center_y = (y0 + y1) // 2
            indicator_size = max(2, cell // 3)  # Scale indicator with cell size
            cv2.rectangle(big_squares_viz, 
                        (center_x - indicator_size, center_y - indicator_size),
                        (center_x + indicator_size, center_y + indicator_size),
                        cluster_colors[s["cluster"]], -1)
            # Add black border around indicator for visibility
            cv2.rectangle(big_squares_viz, 
                        (center_x - indicator_size, center_y - indicator_size),
                        (center_x + indicator_size, center_y + indicator_size),
                        (0, 0, 0), 1)
    
    # Save big squares visualization
    big_squares_path = os.path.join("images/big-squares", p.name)
    cv2.imwrite(big_squares_path, big_squares_viz)
    print(f"   🔳 Saved big squares visualization: {big_squares_path}")

# ────────────────────────────── save JSON ──────────────────────────────
# Remove cluster info from final JSON output
for img_name, squares in results.items():
    for square in squares:
        if 'cluster' in square:
            del square['cluster']

with open("squares.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"✅  wrote {len(results)} image entries → squares.json")
