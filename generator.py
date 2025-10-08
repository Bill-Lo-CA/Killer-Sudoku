from models import Cage
from typing import Optional, List
from constants import UI
from sudoku_full import generate_full_solution

import random
import math

class SudokuGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def puzzle(self):
        b = self.killer_sudoku()
        return self.cage_generator(b), b
    
    def killer_sudoku(self):
        return generate_full_solution(seed=self.seed)
    
    def cage_generator(self, board: List[List[int]]):
        def add(c: Cage, x: int, y: int):
            used_cell.add((x, y))
            c.append(x, y)
            c.add_total(board[x][y])
            return c
        size = UI.GRID_SIZE
        cages = []
        used_cell = set()
        for i in range(size):
            for j in range(size):
                x, y = i, j
                if (x, y) in used_cell: continue
                c = add(Cage(), x, y)
                while self._has_next_cell(len(c)):
                    a, b = self._direction[random.randint(0, 3)]
                    x, y = x + a, y + b
                    if 0 <= x < UI.GRID_SIZE and 0 <= y < UI.GRID_SIZE and (x, y) not in used_cell:
                        c = add(c, x, y)
                    else:
                        x, y = x - a, y - b
                
                cages.append(c)
        
        return cages

    def _has_next_cell(self, curr_cell_num: int) -> bool:
        # Calculate probability of returning True
        if curr_cell_num <= 5:
            # Steep exponential decay from 1 to 5
            # At n=1: ~95% True, at n=5: ~5% True
            prob_true = 0.95 * math.exp(-0.75 * (curr_cell_num - 1))
        else:
            # Gentle logarithmic rise from 5 to 10
            # At n=5: ~5% True, at n=10: ~50% True
            prob_true = 0.05 + 0.45 * math.log(curr_cell_num - 4) / math.log(6)

        return random.random() < prob_true

if __name__ == "__main__":
    g = SudokuGenerator()
    for num in range(10):
        t = 0
        for i in range(100):
            if g._has_next_cell(num):
                t += 1
        print(f"Enter: { num }, Next Cell Percentage: { t }%")
    
    g.cage_generator(g.killer_sudoku())