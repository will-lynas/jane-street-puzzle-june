#!/usr/bin/env python3
"""
annotate_squares.py   –   Draw non-overlapping grid-aligned squares on a
                          45 × 45 grid inside an image.

Usage
-----
    python annotate_squares.py /path/to/image.png

Keys while running
------------------
    • Left-drag              place a square (must be a true square)
    • z                      undo last square
    • s                      save and quit
    • Esc                    quit immediately (nothing written)

Output
------
JSON file is written next to the image with the same name but .json extension.
Format:

{
  "1":  [[x, y], …],
  "2":  [[x, y], …],
  ...
}

Where [x, y] is the *bottom-left* grid coordinate (origin =
bottom-left of whole picture) and the key is the
square side-length in grid cells.
"""
import cv2 as cv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

CELL_COUNT = 45          # grid is 45 × 45
SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# ----------------------------------------------------------------------
class Annotator:
    def __init__(self, img_path: Path):
        self.img_path = img_path
        self.orig     = cv.imread(str(img_path))
        if self.orig is None:
            raise RuntimeError(f'Could not read {img_path}')
        self.h, self.w = self.orig.shape[:2]
        self.cell_w    = self.w / CELL_COUNT
        self.cell_h    = self.h / CELL_COUNT
        self.squares   = []        # list of (left, top, size) in GRID coords
        self._drawing  = False
        self._pt0      = None      # (x_pixel, y_pixel)
        self._current_mouse = None # current mouse position during drag
        self.refresh()

    # ------------- mouse callback ------------------------------------
    def on_mouse(self, event, x, y, flags, _param):
        if event == cv.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._pt0 = (x, y)
            self._current_mouse = (x, y)

        elif event == cv.EVENT_MOUSEMOVE and self._drawing:
            self._current_mouse = (x, y)
            self.refresh()

        elif event == cv.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            self._current_mouse = None
            self.try_add_square(self._pt0, (x, y))
            self.refresh()

    # ------------- helpers -------------------------------------------
    def pix_to_grid(self, x_pix, y_pix):
        col = int(x_pix // self.cell_w)
        row = int(y_pix // self.cell_h)
        # clamp into [0, 44]
        col = max(0, min(CELL_COUNT - 1, col))
        row = max(0, min(CELL_COUNT - 1, row))
        return col, row   # (left-to-right, top-origin)

    def try_add_square(self, p0, p1):
        col0, row0 = self.pix_to_grid(*p0)
        col1, row1 = self.pix_to_grid(*p1)

        left = min(col0, col1)
        top  = min(row0, row1)
        dx   = abs(col1 - col0)
        dy   = abs(row1 - row0)

        if dx != dy:
            print('❌ Not a square – ignored')
            return
        size = dx + 1      # include both endpoints

        if left + size > CELL_COUNT or top + size > CELL_COUNT:
            print('❌ Square extends outside grid – ignored')
            return

        # overlap check
        for (l, t, s) in self.squares:
            if not (left + size <= l or l + s <= left or
                    top  + size <= t or t + s <= top):
                print('❌ Overlaps existing square – ignored')
                return

        self.squares.append((left, top, size))
        x_out = left
        y_out = CELL_COUNT - (top + size)
        print(f'✔️  Added {size}×{size} square at [{x_out}, {y_out}]')

    def undo(self):
        if self.squares:
            popped = self.squares.pop()
            print(f'↩️  Removed last square')
            self.refresh()

    def refresh(self):
        self.view = self.orig.copy()
        # draw every confirmed square in green
        for (l, t, s) in self.squares:
            pt1 = (int((l    ) * self.cell_w), int((t    ) * self.cell_h))
            pt2 = (int((l+s) * self.cell_w)-1, int((t+s) * self.cell_h)-1)
            cv.rectangle(self.view, pt1, pt2, (0,255,0), 2)
        
        # draw preview square in blue while dragging
        if self._drawing and self._pt0 and self._current_mouse:
            col0, row0 = self.pix_to_grid(*self._pt0)
            col1, row1 = self.pix_to_grid(*self._current_mouse)
            
            left = min(col0, col1)
            top  = min(row0, row1)
            dx   = abs(col1 - col0)
            dy   = abs(row1 - row0)
            size = max(dx, dy) + 1  # make it a square using the larger dimension
            
            # clamp to grid bounds
            if left + size > CELL_COUNT:
                left = CELL_COUNT - size
            if top + size > CELL_COUNT:
                top = CELL_COUNT - size
            if left < 0:
                left = 0
            if top < 0:
                top = 0
                
            pt1 = (int(left * self.cell_w), int(top * self.cell_h))
            pt2 = (int((left + size) * self.cell_w) - 1, int((top + size) * self.cell_h) - 1)
            cv.rectangle(self.view, pt1, pt2, (255,0,0), 2)  # blue preview

    # ------------- convert to JSON-ready ------------------------------
    def to_json_entry(self):
        out = defaultdict(list)   # size -> list of [x, y]
        for (left, top, size) in self.squares:
            x_out = left
            y_out = CELL_COUNT - (top + size)
            out[str(size)].append([x_out, y_out])
        # make plain dict for json.dump
        return {k: v for k, v in out.items()}

    def save_json(self):
        json_data = self.to_json_entry()
        json_file = self.img_path.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as fh:
            json.dump(json_data, fh, indent=2, sort_keys=False)
        print(f'💾 Saved {json_file.name}')
        return json_file

# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: python annotate_squares.py /path/to/image.png')
    
    img_path = Path(sys.argv[1]).expanduser().resolve()
    
    if not img_path.exists():
        sys.exit(f'Image file not found: {img_path}')
    
    if img_path.suffix.lower() not in SUPPORTED_EXT:
        sys.exit(f'Unsupported image format. Supported: {", ".join(SUPPORTED_EXT)}')

    ann = Annotator(img_path)
    cv.namedWindow('annotate', cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback('annotate', ann.on_mouse)

    print(f'=== {img_path.name} ===')
    print('Press "s" to save and quit, "z" to undo, "Esc" to quit without saving')
    
    while True:
        cv.imshow('annotate', ann.view)
        key = cv.waitKey(20) & 0xFF

        if key == 27:          # Esc
            print('⛔ Exit requested – nothing saved.')
            cv.destroyAllWindows()
            return

        elif key in (ord('z'), ord('Z')):
            ann.undo()

        elif key in (ord('s'), ord('S')):
            # save and quit
            json_file = ann.save_json()
            print(f'✅ Done. Saved to {json_file.resolve()}')
            cv.destroyAllWindows()
            return

# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()
