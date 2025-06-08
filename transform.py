#!/usr/bin/env python3
"""
grid_perspective.py  –  Perspective-crop a square in an image, then overlay an N×N grid.

Usage (interactive):
    python grid_perspective.py photo.jpg out.png            # click the 4 corners

Usage (non-interactive):
    python grid_perspective.py photo.jpg out.png \
        --pts "123,88  432,92  440,411  115,417"

Extras:
    --cells 45          how many cells per side   (default 45)
    --size  900         output square size (px)  (default 900)
    --color '#00FF00'   grid line colour in hex
    --width 2           grid line thickness (px)
"""

import argparse, ast, sys, cv2
import numpy as np
from pathlib import Path

###############################################################################
# 1. Argument parsing
###############################################################################
p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                            description="Perspective-crop, then overlay a grid.")
p.add_argument("input",  type=Path)
p.add_argument("output", type=Path)
p.add_argument("--cells", type=int, default=45, help="grid cells per side (N)")
p.add_argument("--size",  type=int, default=900, help="final square size in px")
p.add_argument("--color", default="#FF0000", help="grid colour (hex)")
p.add_argument("--width", type=int, default=1, help="grid line thickness")
p.add_argument("--pts",   help='four "x,y" pairs (tl tr br bl) separated by spaces')
args = p.parse_args()

###############################################################################
# 2. Load image
###############################################################################
img = cv2.imread(str(args.input))
if img is None:
    sys.exit(f"❌  Could not read {args.input}")

###############################################################################
# 3. Get four source points
###############################################################################
def hex2bgr(h):                          # "#RRGGBB" → (B,G,R)
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (4,2,0))

if args.pts:
    pts_src = np.array([list(map(float,p.split(",")))
                       for p in args.pts.strip().split()], dtype=np.float32)
    if pts_src.shape != (4,2):
        sys.exit("❌ --pts must supply exactly four x,y pairs")
else:
    # Click-to-select mode
    click_pts = []
    clone = img.copy()
    cv2.namedWindow("select 4 corners — TL ➜ TR ➜ BR ➜ BL")

    def on_mouse(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN and len(click_pts) < 4:
            click_pts.append((x,y))
            cv2.circle(clone,(x,y),4,(0,255,0),-1)
    cv2.setMouseCallback("select 4 corners — TL ➜ TR ➜ BR ➜ BL", on_mouse)

    while True:
        cv2.imshow("select 4 corners — TL ➜ TR ➜ BR ➜ BL", clone)
        if cv2.waitKey(20) & 0xFF == 27:  # ESC to abort
            sys.exit("⏹  Aborted.")
        if len(click_pts) == 4:
            cv2.destroyAllWindows()
            break
    pts_src = np.array(click_pts, dtype=np.float32)

###############################################################################
# 4. Perspective-transform to a square canvas
###############################################################################
N     = args.size
pts_dst = np.array([[0,0], [N-1,0], [N-1,N-1], [0,N-1]], dtype=np.float32)
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped = cv2.warpPerspective(img, M, (N, N))

###############################################################################
# 5. Overlay the grid
###############################################################################
cells = args.cells
step  = N / cells
colour = hex2bgr(args.color)
for i in range(1, cells):
    p = round(i*step)
    cv2.line(warped, (p,0), (p,N-1), colour, args.width)   # vertical
    cv2.line(warped, (0,p), (N-1,p), colour, args.width)    # horizontal

###############################################################################
# 6. Save
###############################################################################
cv2.imwrite(str(args.output), warped)
print(f"✅  Saved perspective-corrected grid image ➜ {args.output}")
