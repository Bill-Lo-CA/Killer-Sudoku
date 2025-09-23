# killer_sudoku.py
import tkinter as tk
from tkinter import messagebox
import time
from dataclasses import dataclass, field

# --- 視覺參數 ---
CELL_SIZE = 60
GRID_SIZE = 9
BOARD_PADDING = 20
LINE_THIN = 1
LINE_THICK = 3

CAGE_LINE_WIDTH = 2        # 籠子線寬（虛線）
CAGE_DASH = (3, 3)         # (dash_length, gap_length)

BG = "#f6eddc"
GRID_COLOR = "#2f2f2f"
HILITE = "#cfe8ff"
SELECT = "#9cc7ff"
FIXED_FG = "#1d4ed8"
USER_FG = "#111111"
ERR_FG = "#d11"
CAGE_SUM_FG = "#6b7280"

FONT_BIG = ("Segoe UI", 20, "bold")
FONT_NOTE = ("Segoe UI", 9)
FONT_UI = ("Segoe UI", 11)

# --- 資料結構 ---
@dataclass
class Cell:
    value: int = 0
    fixed: bool = False
    notes: set = field(default_factory=set)

@dataclass
class Cage:
    total: int
    cells: list  # list of (r, c)

# --- 主程式 ---
class KillerSudokuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Killer Sudoku (Tkinter)")
        self.root.configure(bg=BG)

        w = h = BOARD_PADDING * 2 + CELL_SIZE * GRID_SIZE
        self.canvas = tk.Canvas(root, width=w, height=h, bg=BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

        # UI
        self.btn_undo = tk.Button(root, text="Undo", command=self.undo, font=FONT_UI)
        self.btn_clear = tk.Button(root, text="Clear", command=self.clear_cell, font=FONT_UI)
        self.btn_pencil = tk.Button(root, text="Pencil: OFF", command=self.toggle_pencil, font=FONT_UI)
        self.btn_reset = tk.Button(root, text="Reset", command=self.reset, font=FONT_UI)
        self.btn_hint = tk.Button(root, text="Hint (stub)", command=self.hint, font=FONT_UI)

        self.btn_undo.grid(row=1, column=0, padx=6, pady=(0,10), sticky="ew")
        self.btn_clear.grid(row=1, column=1, padx=6, pady=(0,10), sticky="ew")
        self.btn_pencil.grid(row=1, column=2, padx=6, pady=(0,10), sticky="ew")
        self.btn_reset.grid(row=1, column=3, padx=6, pady=(0,10), sticky="ew")
        self.btn_hint.grid(row=1, column=4, padx=6, pady=(0,10), sticky="ew")

        self.status = tk.Label(root, text="Mistakes: 0   Time: 00:00", anchor="w", bg=BG, font=FONT_UI)
        self.status.grid(row=2, column=0, columnspan=5, sticky="ew", padx=10, pady=(0,10))

        # 狀態
        self.board = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.cages = self._sample_cages()
        self.fixed_starters = self._sample_starters()
        self.apply_starters()

        self.selected = (0, 0)
        self.pencil_mode = False
        self.mistakes = 0
        self.start_time = time.time()
        self.undo_stack = []

        # 綁定
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Key>", self.on_key)

        self.redraw()
        self.update_clock()

    # ---- 範例盤面 ----
    def _sample_starters(self):
        # (row, col): value  — 非截圖原題，僅示意
        return {
            (1, 2): 7,
            (3, 0): 6,
            (4, 2): 6,
            (3, 4): 1,
            (3, 5): 9,
            (3, 7): 8,
            (3, 8): 4,
            (7, 1): 3,
            (8, 7): 6,
        }

    def _sample_cages(self):
        # 示意籠子；你可依實際關卡調整
        return [
            Cage(13, [(0,0),(0,1),(1,0)]),
            Cage(20, [(0,2),(0,3),(1,3)]),
            Cage(14, [(0,4),(0,5),(1,5)]),
            Cage(11, [(0,6),(1,6)]),
            Cage(27, [(1,1),(1,2),(2,0),(2,1)]),
            Cage(8,  [(2,3),(2,4)]),
            Cage(19, [(2,5),(2,6),(1,4)]),
            Cage(15, [(0,7),(0,8),(1,8)]),
            Cage(17, [(1,7),(2,7)]),
            Cage(12, [(3,2),(3,3)]),
            Cage(10, [(3,4),(3,6)]),
            Cage(16, [(4,4),(4,5),(5,4)]),
            Cage(13, [(4,1),(5,1)]),
            Cage(10, [(4,2),(5,2)]),
            Cage(21, [(6,4),(6,5),(6,6)]),
            Cage(14, [(6,3),(7,3)]),
            Cage(15, [(6,7),(7,7),(8,7)]),
            Cage(11, [(7,5),(8,5)]),
            Cage(20, [(6,0),(6,1),(7,0)]),
            Cage(12, [(6,2),(6,3)]),
            Cage(13, [(7,2),(7,3)]),
            Cage(10, [(7,4),(8,4)]),
            Cage(6,  [(8,6)]),
            Cage(10, [(8,8),(7,8)]),
        ]

    def apply_starters(self):
        for (r, c), v in self.fixed_starters.items():
            cell = self.board[r][c]
            cell.value = v
            cell.fixed = True

    # ---- 繪圖 ----
    def redraw(self):
        self.canvas.delete("all")
        x0 = y0 = BOARD_PADDING

        # 高亮區
        sr, sc = self.selected
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x = x0 + c * CELL_SIZE
                y = y0 + r * CELL_SIZE
                if r == sr or c == sc or (r//3, c//3)==(sr//3, sc//3):
                    self.canvas.create_rectangle(x, y, x+CELL_SIZE, y+CELL_SIZE, fill=HILITE, outline="")
        # 選中
        x = x0 + sc * CELL_SIZE
        y = y0 + sr * CELL_SIZE
        self.canvas.create_rectangle(x, y, x+CELL_SIZE, y+CELL_SIZE, fill=SELECT, outline="")

        # 1) 先畫網格線
        for i in range(GRID_SIZE+1):
            w = LINE_THICK if i % 3 == 0 else LINE_THIN
            # 橫線
            self.canvas.create_line(
                x0, y0 + i*CELL_SIZE, x0 + GRID_SIZE*CELL_SIZE, y0 + i*CELL_SIZE,
                width=w, fill=GRID_COLOR
            )
            # 直線
            self.canvas.create_line(
                x0 + i*CELL_SIZE, y0, x0 + i*CELL_SIZE, y0 + GRID_SIZE*CELL_SIZE,
                width=w, fill=GRID_COLOR
            )

        # 2) 再畫籠子（虛線會在網格之上）
        self.draw_cages()

        # 3) 數字/筆記
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = self.board[r][c]
                cx = x0 + c * CELL_SIZE + CELL_SIZE/2
                cy = y0 + r * CELL_SIZE + CELL_SIZE/2
                if cell.value:
                    fg = FIXED_FG if cell.fixed else USER_FG
                    self.canvas.create_text(cx, cy, text=str(cell.value), font=FONT_BIG, fill=fg)
                elif cell.notes:
                    self.draw_notes(r, c, cell.notes)

    def draw_notes(self, r, c, notes):
        x0 = BOARD_PADDING + c * CELL_SIZE
        y0 = BOARD_PADDING + r * CELL_SIZE
        for n in range(1, 10):
            if n in notes:
                sub_r = (n-1)//3
                sub_c = (n-1)%3
                px = x0 + (sub_c+0.5)*(CELL_SIZE/3)
                py = y0 + (sub_r+0.5)*(CELL_SIZE/3)
                self.canvas.create_text(px, py, text=str(n), font=FONT_NOTE, fill="#374151")

    def draw_cages(self):
        x0 = y0 = BOARD_PADDING
        in_cage = {}
        for cage in self.cages:
            for (r,c) in cage.cells:
                in_cage[(r,c)] = cage

        for cage in self.cages:
            # 和數標籤：籠子最靠左上角的 cell
            top_left = min(cage.cells)
            tx = x0 + top_left[1]*CELL_SIZE + 4
            ty = y0 + top_left[0]*CELL_SIZE + 4
            self.canvas.create_text(tx, ty, anchor="nw", text=str(cage.total),
                                    font=("Segoe UI", 10, "bold"), fill=CAGE_SUM_FG)

            for (r,c) in cage.cells:
                x = x0 + c * CELL_SIZE
                y = y0 + r * CELL_SIZE
                # 上
                if (r-1, c) not in in_cage:
                    self.canvas.create_line(x, y, x+CELL_SIZE, y,
                                            dash=CAGE_DASH, width=CAGE_LINE_WIDTH)
                # 下
                if (r+1, c) not in in_cage:
                    self.canvas.create_line(x, y+CELL_SIZE, x+CELL_SIZE, y+CELL_SIZE,
                                            dash=CAGE_DASH, width=CAGE_LINE_WIDTH)
                # 左
                if (r, c-1) not in in_cage:
                    self.canvas.create_line(x, y, x, y+CELL_SIZE,
                                            dash=CAGE_DASH, width=CAGE_LINE_WIDTH)
                # 右
                if (r, c+1) not in in_cage:
                    self.canvas.create_line(x+CELL_SIZE, y, x+CELL_SIZE, y+CELL_SIZE,
                                            dash=CAGE_DASH, width=CAGE_LINE_WIDTH)

    # ---- 輸入處理 ----
    def on_click(self, event):
        r, c = self.xy_to_rc(event.x, event.y)
        if r is not None:
            self.selected = (r, c)
            self.redraw()

    def on_key(self, event):
        if event.keysym in ("BackSpace", "Delete"):
            self.clear_cell(); return
        if event.char and event.char.isdigit():
            d = int(event.char)
            if 1 <= d <= 9:
                self.enter_number(d); return
        if event.keysym.lower() == "p":
            self.toggle_pencil(); return

    def xy_to_rc(self, x, y):
        x0 = y0 = BOARD_PADDING
        if not (x0 <= x <= x0 + CELL_SIZE*GRID_SIZE and y0 <= y <= y0 + CELL_SIZE*GRID_SIZE):
            return None, None
        c = (x - x0) // CELL_SIZE
        r = (y - y0) // CELL_SIZE
        return int(r), int(c)

    # ---- 操作 ----
    def push_undo(self):
        snapshot = [[(cell.value, set(cell.notes), cell.fixed) for cell in row] for row in self.board]
        self.undo_stack.append(snapshot)

    def pop_undo(self):
        if self.undo_stack:
            snap = self.undo_stack.pop()
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    v, notes, fixed = snap[r][c]
                    self.board[r][c].value = v
                    self.board[r][c].notes = set(notes)
                    self.board[r][c].fixed = fixed

    def undo(self):
        self.pop_undo()
        self.redraw()

    def toggle_pencil(self):
        self.pencil_mode = not self.pencil_mode
        self.btn_pencil.config(text=f"Pencil: {'ON' if self.pencil_mode else 'OFF'}")

    def clear_cell(self):
        r, c = self.selected
        cell = self.board[r][c]
        if cell.fixed: return
        self.push_undo()
        if cell.value:
            cell.value = 0
        else:
            cell.notes.clear()
        self.redraw()

    def enter_number(self, d):
        r, c = self.selected
        cell = self.board[r][c]
        if cell.fixed: return
        self.push_undo()
        if self.pencil_mode:
            if d in cell.notes:
                cell.notes.remove(d)
            else:
                if cell.value:
                    cell.value = 0
                cell.notes.add(d)
        else:
            cell.notes.clear()
            cell.value = d
            if self.has_conflict(r, c) or self.cage_conflict(r, c):
                self.mistakes += 1
                self.canvas.after(0, self.flash_error, r, c)
        self.redraw()
        if self.is_complete() and self.all_cages_ok():
            messagebox.showinfo(
                "Congrats!",
                f"Puzzle solved!\nTime: {self.elapsed_str()}  Mistakes: {self.mistakes}"
            )

    def flash_error(self, r, c):
        x0 = BOARD_PADDING + c * CELL_SIZE + CELL_SIZE/2
        y0 = BOARD_PADDING + r * CELL_SIZE + CELL_SIZE/2
        self.canvas.create_text(x0, y0, text=str(self.board[r][c].value),
                                font=FONT_BIG, fill=ERR_FG, tag="err")
        self.canvas.after(300, lambda: self.canvas.delete("err"))

    # ---- 規則檢查 ----
    def has_conflict(self, r, c):
        v = self.board[r][c].value
        if v == 0: return False
        # row / col
        for i in range(GRID_SIZE):
            if i != c and self.board[r][i].value == v: return True
            if i != r and self.board[i][c].value == v: return True
        # box
        br = (r//3)*3
        bc = (c//3)*3
        for rr in range(br, br+3):
            for cc in range(bc, bc+3):
                if (rr,cc)!=(r,c) and self.board[rr][cc].value == v:
                    return True
        return False

    def cage_of(self, r, c):
        for cage in self.cages:
            if (r,c) in cage.cells:
                return cage
        return None

    def cage_conflict(self, r, c):
        cage = self.cage_of(r, c)
        if not cage: return False
        vals = [self.board[rr][cc].value for (rr,cc) in cage.cells if self.board[rr][cc].value != 0]
        if len(vals) != len(set(vals)):  # 籠子內不重複
            return True
        s = sum(vals)
        if s > cage.total:              # 局部和不可超過
            return True
        if all(self.board[rr][cc].value != 0 for (rr,cc) in cage.cells) and s != cage.total:
            return True
        return False

    def all_cages_ok(self):
        for cage in self.cages:
            vals = [self.board[r][c].value for (r,c) in cage.cells]
            if 0 in vals:
                if sum(v for v in vals if v) > cage.total:
                    return False
            else:
                if sum(vals) != cage.total or len(vals) != len(set(vals)):
                    return False
        return True

    def is_complete(self):
        return all(self.board[r][c].value != 0 for r in range(GRID_SIZE) for c in range(GRID_SIZE))

    # ---- 其他 ----
    def reset(self):
        self.board = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.apply_starters()
        self.selected = (0,0)
        self.mistakes = 0
        self.undo_stack.clear()
        self.start_time = time.time()
        self.redraw()

    def hint(self):
        messagebox.showinfo("Hint", "這裡提供框架；之後可接上解題器/提示邏輯。")

    def update_clock(self):
        self.status.config(text=f"Mistakes: {self.mistakes}   Time: {self.elapsed_str()}")
        self.root.after(500, self.update_clock)

    def elapsed_str(self):
        sec = int(time.time() - self.start_time)
        return f"{sec//60:02d}:{sec%60:02d}"

def main():
    root = tk.Tk()
    app = KillerSudokuApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
