# Jane Street June 2025 puzzle

A Python solver for the nine tiling boards in [Some Ones, Somewhere](https://www.janestreet.com/puzzles/some-ones-somewhere-index/).

The solution was originally written on 16 June 2025 and cleaned up later.

The repo uses OpenCV to straighten the puzzle photos and OR-Tools to complete each 45×45 board from manually marked tiles. The solver reports the position of the 1×1 tile, which encodes two letters of the answer.

## Usage

To run the solver on one board:

```sh
uv run solve.py top-left
```

The nine positions are in [`results.txt`](results.txt).
