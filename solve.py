from ortools.sat.python import cp_model
import json
import sys
import os

def build_instance(json_filename=None):
    """Return (sides, preplaced).
       sides      : [1, 2, 2, 3, 3, 3, …, 9]   (length 45)
       preplaced  : list[tuple[int,int,int]]  each = (size, x, y)
                    (x, y) is lower-left corner in 0-based coords.
       If json_filename is provided, load preplaced squares from that file.
    """
    # 45×45 container is fixed
    sides = [k for k in range(1, 10) for _ in range(k)]

    preplaced = []
    
    if json_filename:
        json_path = os.path.join("json", json_filename)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert from JSON format to preplaced format
        for size_str, coordinates in data.items():
            size = int(size_str)
            for x, y in coordinates:
                preplaced.append((size, x, y))

    return sides, preplaced


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    """Callback to print solutions as they are found."""
    
    def __init__(self, x_vars, y_vars, sides):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.sides = sides
        self.solution_count = 0
    
    def on_solution_callback(self):
        self.solution_count += 1
        print(f"\nSolution {self.solution_count}:")
        for i in range(len(self.sides)):
            x_val = self.Value(self.x_vars[i])
            y_val = self.Value(self.y_vars[i])
            s = self.sides[i]
            print(f"• {s}×{s} at ({x_val}, {y_val})")


def pack_fixed_container(B: int, sides, preplaced):
    """Try to place all squares inside B×B respecting the pre-placements.
       Prints all solutions as they are found.
       Returns the number of solutions found.
    """
    n = len(sides)
    model = cp_model.CpModel()

    # 1 ▸ decision variables
    x = [model.NewIntVar(0, B - s, f"x{i}") for i, s in enumerate(sides)]
    y = [model.NewIntVar(0, B - s, f"y{i}") for i, s in enumerate(sides)]

    # 2 ▸ respect user-fixed squares
    #     greedy assignment: first unused square of matching size is fixed
    used = [False] * n
    for size, fx, fy in preplaced:
        for i, s in enumerate(sides):
            if not used[i] and s == size:
                used[i] = True
                model.Add(x[i] == fx)
                model.Add(y[i] == fy)
                break
        else:
            raise ValueError(f"No unused square of size {size} left to fix at {(fx, fy)}")

    # 3 ▸ interval views
    x_ints, y_ints = [], []
    for i, s in enumerate(sides):
        x_ints.append(model.NewIntervalVar(x[i], s, x[i] + s, f"x_int{i}"))
        y_ints.append(model.NewIntervalVar(y[i], s, y[i] + s, f"y_int{i}"))

    # 4 ▸ global non-overlap
    model.AddNoOverlap2D(x_ints, y_ints)

    # 5 ▸ symmetry break: anchor one 9×9 in a corner if unfixed
    try:
        anchor = sides.index(9)
        if not (x[anchor].HasValue() and y[anchor].HasValue()):
            model.Add(x[anchor] == 0)
            model.Add(y[anchor] == 0)
    except ValueError:
        pass  # should never happen (we have nine 9×9s)

    # 6 ▸ solve for all solutions
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_search_workers = 8
    solver.parameters.enumerate_all_solutions = True
    
    solution_collector = SolutionCollector(x, y, sides)
    status = solver.SearchForAllSolutions(model, solution_collector)

    return solution_collector.solution_count


if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_filename = sys.argv[1]
        if not json_filename.endswith('.json'):
            json_filename += '.json'
    else:
        json_filename = None
    
    B = 45
    sides, preplaced = build_instance(json_filename)
    
    if json_filename:
        print(f"Loading preplaced squares from: {json_filename}")
        print(f"Found {len(preplaced)} preplaced squares")
    
    print(f"Searching for all solutions inside {B}×{B}...")
    
    solution_count = pack_fixed_container(B, sides, preplaced)
    
    if solution_count == 0:
        print("No feasible packing with those fixed squares.")
    else:
        print(f"\nFound {solution_count} total solution(s).")

