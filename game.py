import time
from typing import List, Tuple
from constants import GRID_SIZE
from models import Cell, Cage

Snapshot = List[List[Tuple[int, set, bool]]]

class Game:
    """純邏輯層（不依賴 Tk）：盤面、規則、undo、計時等。"""
    def __init__(self, starters: dict, cages: List[Cage], grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self.board: List[List[Cell]] = [[Cell() for _ in range(grid_size)] for _ in range(grid_size)]
        self.cages = cages
        self.fixed_starters = starters

        self.selected = (0, 0)
        self.pencil_mode = False
        self.mistakes = 0
        self.start_time = time.time()
        self.undo_stack: List[Snapshot] = []

        self.apply_starters()

    # ---- 狀態/快照 ----
    def apply_starters(self):
        for (r, c), v in self.fixed_starters.items():
            cell = self.board[r][c]
            cell.value = v
            cell.fixed = True

    def push_undo(self):
        snap: Snapshot = [[(cell.value, set(cell.notes), cell.fixed) for cell in row] for row in self.board]
        self.undo_stack.append(snap)

    def pop_undo(self):
        if not self.undo_stack:
            return
        snap = self.undo_stack.pop()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                v, notes, fixed = snap[r][c]
                self.board[r][c].value = v
                self.board[r][c].notes = set(notes)
                self.board[r][c].fixed = fixed

    # ---- 互動 ----
    def select(self, r: int, c: int):
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            self.selected = (r, c)

    def toggle_pencil(self):
        self.pencil_mode = not self.pencil_mode
        return self.pencil_mode

    def clear_cell(self):
        r, c = self.selected
        cell = self.board[r][c]
        if cell.fixed:
            return
        self.push_undo()
        if cell.value:
            cell.value = 0
        else:
            cell.notes.clear()

    def enter_number(self, d: int):
        """回傳 (conflict: bool, solved: bool)"""
        r, c = self.selected
        cell = self.board[r][c]
        if cell.fixed:
            return False, self.is_solved()

        self.push_undo()
        if self.pencil_mode:
            if d in cell.notes:
                cell.notes.remove(d)
            else:
                if cell.value:
                    cell.value = 0
                cell.notes.add(d)
            return False, self.is_solved()

        # 非筆記：填入值
        cell.notes.clear()
        cell.value = d
        conflict = self.has_conflict(r, c) or self.cage_conflict(r, c)
        if conflict:
            self.mistakes += 1
        return conflict, self.is_solved()

    # ---- 規則檢查 ----
    def has_conflict(self, r: int, c: int) -> bool:
        v = self.board[r][c].value
        if v == 0:
            return False
        # row / col
        for i in range(self.grid_size):
            if i != c and self.board[r][i].value == v:
                return True
            if i != r and self.board[i][c].value == v:
                return True
        # box
        br = (r // 3) * 3
        bc = (c // 3) * 3
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if (rr, cc) != (r, c) and self.board[rr][cc].value == v:
                    return True
        # 新增 同一個BOX中數字超過視為confilct
        return False

    def cage_of(self, r: int, c: int):
        for cage in self.cages:
            if (r, c) in cage.cells:
                return cage
        return None

    def cage_conflict(self, r: int, c: int) -> bool:
        cage = self.cage_of(r, c)
        if not cage:
            return False
        vals = [self.board[rr][cc].value for (rr, cc) in cage.cells if self.board[rr][cc].value != 0]
        if len(vals) != len(set(vals)):  # 籠子內不重複
            return True
        s = sum(vals)
        if s > cage.total:              # 局部和不可超過
            return True
        if all(self.board[rr][cc].value != 0 for (rr, cc) in cage.cells) and s != cage.total:
            return True
        return False

    def all_cages_ok(self) -> bool:
        for cage in self.cages:
            vals = [self.board[r][c].value for (r, c) in cage.cells]
            if 0 in vals:
                if sum(v for v in vals if v) > cage.total:
                    return False
            else:
                if sum(vals) != cage.total or len(vals) != len(set(vals)):
                    return False
        return True

    def is_complete(self) -> bool:
        return all(self.board[r][c].value != 0 for r in range(self.grid_size) for c in range(self.grid_size))

    def is_solved(self) -> bool:
        return self.is_complete() and self.all_cages_ok()

    # ---- 其他 ----
    def reset(self):
        self.board = [[Cell() for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.apply_starters()
        self.selected = (0, 0)
        self.mistakes = 0
        self.undo_stack.clear()
        self.start_time = time.time()

    def elapsed_str(self) -> str:
        sec = int(time.time() - self.start_time)
        return f"{sec//60:02d}:{sec%60:02d}"