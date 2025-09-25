import tkinter as tk
from tkinter import messagebox
from constants import (
    CELL_SIZE, GRID_SIZE, BOARD_PADDING, LINE_THIN, LINE_THICK,
    CAGE_LINE_WIDTH, CAGE_LINE_SHIFT, CAGE_DASH,
    BG, GRID_COLOR, HILITE, SELECT, FIXED_FG, USER_FG, ERR_FG, CAGE_SUM_FG,
    FONT_BIG, FONT_NOTE, FONT_UI
)
from game import Game

class KillerSudokuApp:
    def __init__(self, root: tk.Tk, game: Game):
        self.root = root
        self.game = game

        self.root.title("Killer Sudoku (Tkinter)")
        self.root.configure(bg=BG)

        w = h = BOARD_PADDING * 2 + CELL_SIZE * GRID_SIZE
        self.canvas = tk.Canvas(root, width=w, height=h, bg=BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

        # UI
        self.btn_undo   = tk.Button(root, text="Undo",         command=self.undo,          font=FONT_UI)
        self.btn_clear  = tk.Button(root, text="Clear",        command=self.clear_cell,    font=FONT_UI)
        self.btn_pencil = tk.Button(root, text="Pencil: OFF",  command=self.toggle_pencil, font=FONT_UI)
        self.btn_reset  = tk.Button(root, text="Reset",        command=self.reset,         font=FONT_UI)
        self.btn_hint   = tk.Button(root, text="Hint (stub)",  command=self.hint,          font=FONT_UI)

        self.btn_undo.grid(  row=1, column=0, padx=6, pady=(0,10), sticky="ew")
        self.btn_clear.grid( row=1, column=1, padx=6, pady=(0,10), sticky="ew")
        self.btn_pencil.grid(row=1, column=2, padx=6, pady=(0,10), sticky="ew")
        self.btn_reset.grid( row=1, column=3, padx=6, pady=(0,10), sticky="ew")
        self.btn_hint.grid(  row=1, column=4, padx=6, pady=(0,10), sticky="ew")

        self.status = tk.Label(root, text="Mistakes: 0   Time: 00:00", anchor="w", bg=BG, font=FONT_UI)
        self.status.grid(row=2, column=0, columnspan=5, sticky="ew", padx=10, pady=(0,10))

        # 綁定
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Key>", self.on_key)

        self.redraw()
        self.update_clock()

    # ---- 事件 ----
    def on_click(self, event):
        r, c = self.xy_to_rc(event.x, event.y)
        if r is not None:
            self.game.select(r, c)
            self.redraw()

    def on_key(self, event):
        if event.keysym in ("BackSpace", "Delete"):
            self.clear_cell(); return
        if event.char and event.char.isdigit():
            d = int(event.char)
            if 1 <= d <= 9:
                conflict, solved = self.game.enter_number(d)
                if conflict:
                    self.flash_error(*self.game.selected)
                self.redraw()
                if solved:
                    messagebox.showinfo("Congrats!", f"Puzzle solved!\nTime: {self.game.elapsed_str()}  Mistakes: {self.game.mistakes}")
                return
        if event.keysym.lower() == "p":
            self.toggle_pencil(); return

    # ---- 工具 ----
    def xy_to_rc(self, x, y):
        x0 = y0 = BOARD_PADDING
        if not (x0 <= x <= x0 + CELL_SIZE*GRID_SIZE and y0 <= y <= y0 + CELL_SIZE*GRID_SIZE):
            return None, None
        c = (x - x0) // CELL_SIZE
        r = (y - y0) // CELL_SIZE
        return int(r), int(c)

    # ---- 操作 ----
    def undo(self):
        self.game.pop_undo()
        self.redraw()

    def toggle_pencil(self):
        on = self.game.toggle_pencil()
        self.btn_pencil.config(text=f"Pencil: {'ON' if on else 'OFF'}")

    def clear_cell(self):
        self.game.clear_cell()
        self.redraw()

    def reset(self):
        self.game.reset()
        self.btn_pencil.config(text="Pencil: OFF")
        self.redraw()

    def hint(self):
        messagebox.showinfo("Hint", "這裡提供框架；之後可接上解題器/提示邏輯。")

    # ---- 視覺 ----
    def redraw(self):
        self.canvas.delete("all")
        x0 = y0 = BOARD_PADDING

        # 高亮區
        sr, sc = self.game.selected
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

        # 網格線
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

        # 籠子（虛線在網格之上）
        self.draw_cages()

        # 數字/筆記
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = self.game.board[r][c]
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
        for cage in self.game.cages:
            in_cage = {}
            for (r,c) in cage.cells:
                in_cage[(r,c)] = cage
            # 和數標籤：籠子最靠左上角的 cell（以 row, col lexicographic 最小視為左上）
            top_left_x, top_left_y = min(cage.cells)
            tx = x0 + top_left_x * CELL_SIZE + 4
            ty = y0 + top_left_y * CELL_SIZE + 4
            self.canvas.create_text(ty, tx, anchor="nw", text=str(cage.total),
                                    font=("Segoe UI", 10, "bold"), fill=CAGE_SUM_FG)
            for (r,c) in cage.cells:
                x = x0 + c * CELL_SIZE
                y = y0 + r * CELL_SIZE
                # 上
                if (r-1, c) not in in_cage:
                    self.canvas.create_line(x, y + CAGE_LINE_SHIFT, x + CELL_SIZE, y + CAGE_LINE_SHIFT,
                                            dash=CAGE_DASH, width=CAGE_LINE_WIDTH)
                # 下
                if (r+1, c) not in in_cage:
                    self.canvas.create_line(x, y + CELL_SIZE - CAGE_LINE_SHIFT, x + CELL_SIZE, y + CELL_SIZE - CAGE_LINE_SHIFT,
                                            dash=CAGE_DASH, width=CAGE_LINE_WIDTH)
                # 左
                if (r, c-1) not in in_cage:
                    self.canvas.create_line(x + CAGE_LINE_SHIFT, y, x + CAGE_LINE_SHIFT, y + CELL_SIZE,
                                            dash=CAGE_DASH, width=CAGE_LINE_WIDTH)
                # 右
                if (r, c+1) not in in_cage:
                    self.canvas.create_line(x + CELL_SIZE - CAGE_LINE_SHIFT, y, x + CELL_SIZE - CAGE_LINE_SHIFT, y + CELL_SIZE,
                                            dash=CAGE_DASH, width=CAGE_LINE_WIDTH)

    def flash_error(self, r, c):
        x0 = BOARD_PADDING + c * CELL_SIZE + CELL_SIZE/2
        y0 = BOARD_PADDING + r * CELL_SIZE + CELL_SIZE/2
        v = self.game.board[r][c].value
        self.canvas.create_text(x0, y0, text=str(v), font=FONT_BIG, fill=ERR_FG, tag="err")
        self.canvas.after(300, lambda: self.canvas.delete("err"))

    # ---- 狀態 ----
    def update_clock(self):
        self.status.config(text=f"Mistakes: {self.game.mistakes}   Time: {self.game.elapsed_str()}")
        self.root.after(500, self.update_clock)