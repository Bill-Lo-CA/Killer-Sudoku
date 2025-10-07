import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
from constants import UI, COLOR, FONT
from game import Game

class KillerSudokuApp:
    def __init__(self, root: tk.Tk, game: Game):
        self.root = root
        self.game = game

        self.root.title("Killer Sudoku")
        self.root.configure(bg=COLOR.BG)

        w = h = UI.BOARD_PADDING * 2 + UI.CELL_SIZE * UI.GRID_SIZE
        self.canvas = tk.Canvas(root, width=w, height=h, bg=COLOR.BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

        # UI
        self.btn_undo   = tk.Button(root, text = "Undo",         command = self.undo,          font = FONT.FONT_UI)
        self.btn_clear  = tk.Button(root, text = "Clear",        command = self.clear_cell,    font = FONT.FONT_UI)
        self.btn_pencil = tk.Button(root, text = "Pencil: OFF",  command = self.toggle_pencil, font = FONT.FONT_UI)
        self.btn_reset  = tk.Button(root, text = "Reset",        command = self.reset,         font = FONT.FONT_UI)
        self.btn_hint   = tk.Button(root, text = "Hint (stub)",  command = self.hint,          font = FONT.FONT_UI)

        self.btn_undo.grid(  row = 1, column = 0, padx = 6, pady = (0, 10), sticky = "ew")
        self.btn_clear.grid( row = 1, column = 1, padx = 6, pady = (0, 10), sticky = "ew")
        self.btn_pencil.grid(row = 1, column = 2, padx = 6, pady = (0, 10), sticky = "ew")
        self.btn_reset.grid( row = 1, column = 3, padx = 6, pady = (0, 10), sticky = "ew")
        self.btn_hint.grid(  row = 1, column = 4, padx = 6, pady = (0, 10), sticky = "ew")

        self.status = tk.Label(root, text="Mistakes: 0   Time: 00:00", anchor="w", bg=COLOR.BG, font=FONT.FONT_UI)
        self.status.grid(row=2, column=0, columnspan=5, sticky="ew", padx=10, pady=(0,10))

        self.cage_data = [(min(cage.cells), str(cage.total)) for cage in self.game.cages]

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
        x0 = y0 = UI.BOARD_PADDING
        if not (x0 <= x <= x0 + UI.CELL_SIZE*UI.GRID_SIZE and y0 <= y <= y0 + UI.CELL_SIZE*UI.GRID_SIZE):
            return None, None
        c = (x - x0) // UI.CELL_SIZE
        r = (y - y0) // UI.CELL_SIZE
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

    def all_hint(self):
        return

    # ---- 視覺 ----
    def redraw(self):
        self.canvas.delete("all")
        x0 = y0 = UI.BOARD_PADDING

        # 高亮區
        sr, sc = self.game.selected
        for r in range(UI.GRID_SIZE):
            for c in range(UI.GRID_SIZE):
                x = x0 + c * UI.CELL_SIZE
                y = y0 + r * UI.CELL_SIZE
                if r == sr or c == sc or (r // 3, c // 3) == (sr // 3, sc // 3):
                    self.canvas.create_rectangle(x, y, x + UI.CELL_SIZE, y + UI.CELL_SIZE,
                                                 fill=COLOR.HILITE, outline="")

        # 選中
        x = x0 + sc * UI.CELL_SIZE
        y = y0 + sr * UI.CELL_SIZE
        self.canvas.create_rectangle(x, y, x + UI.CELL_SIZE, y + UI.CELL_SIZE,
                                     fill=COLOR.SELECT, outline="")

        # 網格線
        for i in range(UI.GRID_SIZE+1):
            w = UI.LINE_THICK if i % 3 == 0 else UI.LINE_THIN
            # 橫線
            self.canvas.create_line(
                x0, y0 + i * UI.CELL_SIZE, x0 + UI.GRID_SIZE * UI.CELL_SIZE, y0 + i * UI.CELL_SIZE,
                width=w, fill=COLOR.GRID_COLOR
            )
            # 直線
            self.canvas.create_line(
                x0 + i * UI.CELL_SIZE, y0, x0 + i * UI.CELL_SIZE, y0 + UI.GRID_SIZE * UI.CELL_SIZE,
                width=w, fill=COLOR.GRID_COLOR
            )

        # 籠子（虛線在網格之上）
        self.draw_cages()

        # 數字/筆記
        for r in range(UI.GRID_SIZE):
            for c in range(UI.GRID_SIZE):
                cell = self.game.board[r][c]
                cx = x0 + c * UI.CELL_SIZE + UI.CELL_SIZE / 2
                cy = y0 + r * UI.CELL_SIZE + UI.CELL_SIZE / 2
                if cell.value:
                    fg = COLOR.FIXED_FG if cell.fixed else COLOR.USER_FG
                    self.canvas.create_text(cx, cy, text=str(cell.value), font=FONT.FONT_BIG, fill=fg)
                elif cell.notes:
                    self.draw_notes(r, c, cell.notes)

    def draw_notes(self, r, c, notes):
        x0 = UI.BOARD_PADDING + c * UI.CELL_SIZE
        y0 = UI.BOARD_PADDING + r * UI.CELL_SIZE
        for n in range(1, 10):
            if n in notes:
                sub_r = (n - 1) // 3
                sub_c = (n - 1) % 3
                px = x0 + (sub_c + 0.5) * (UI.CELL_SIZE / 3)
                py = y0 + (sub_r + 0.5) * (UI.CELL_SIZE / 3)
                self.canvas.create_text(px, py, text=str(n), font=FONT.FONT_NOTE, fill=COLOR.NOTE_COLOR)

    def draw_cages(self):
        x0 = y0 = UI.BOARD_PADDING
        for cage in self.game.cages:
            in_cage = { (r, c): cage for (r, c) in cage.cells }

            for (r, c) in cage.cells:
                x = x0 + c * UI.CELL_SIZE
                y = y0 + r * UI.CELL_SIZE
                # 上
                if (r-1, c) not in in_cage:
                    self.canvas.create_line(
                        x, y + UI.CAGE_LINE_SHIFT, x + UI.CELL_SIZE, y + UI.CAGE_LINE_SHIFT,
                        dash=UI.CAGE_DASH, width=UI.CAGE_LINE_WIDTH
                    )
                # 下
                if (r+1, c) not in in_cage:
                    self.canvas.create_line(
                        x, y + UI.CELL_SIZE - UI.CAGE_LINE_SHIFT, x + UI.CELL_SIZE, y + UI.CELL_SIZE - UI.CAGE_LINE_SHIFT,
                        dash=UI.CAGE_DASH, width=UI.CAGE_LINE_WIDTH
                    )
                # 左
                if (r, c-1) not in in_cage:
                    self.canvas.create_line(
                        x + UI.CAGE_LINE_SHIFT, y, x + UI.CAGE_LINE_SHIFT, y + UI.CELL_SIZE,
                        dash=UI.CAGE_DASH, width=UI.CAGE_LINE_WIDTH
                    )
                # 右
                if (r, c+1) not in in_cage:
                    self.canvas.create_line(
                        x + UI.CELL_SIZE - UI.CAGE_LINE_SHIFT, y, x + UI.CELL_SIZE - UI.CAGE_LINE_SHIFT, y + UI.CELL_SIZE,
                        dash=UI.CAGE_DASH, width=UI.CAGE_LINE_WIDTH
                    )
        self.draw_total()

    def draw_total(self):
        x0 = y0 = UI.BOARD_PADDING
        sr, sc = self.game.selected
        for ((top_left_x, top_left_y), text) in self.cage_data:
            # 和數標籤：籠子最靠左上角的 cell（以 row, col lexicographic 最小視為左上）
            tx = x0 + top_left_x * UI.CELL_SIZE
            ty = y0 + top_left_y * UI.CELL_SIZE

            # 用 UI 字體為基底建立粗體以量測（這裡固定 10pt 粗體也可）
            # 若想嚴格沿用 FONT.FONT_UI 的家族與大小，可解開下行並視需求調整 size
            font_obj = tkfont.Font(family="Segoe UI", size=10, weight="bold")

            # 固定正方形邊長（你也可改成動態量測）
            side = 17

            x, y = ty + UI.CAGE_PAD, tx + UI.CAGE_PAD

            # 背景色：選中格 > 同列/同欄/同宮 > 一般
            color = (
                COLOR.SELECT
                if top_left_x == sr and top_left_y == sc
                else COLOR.HILITE
                if (top_left_x == sr or top_left_y == sc or
                    (top_left_x // 3, top_left_y // 3) == (sr // 3, sc // 3))
                else COLOR.BG
            )

            # 先畫底
            self.canvas.create_rectangle(
                x, y, x + side, y + side,
                fill=color, outline=""
            )

            # 再畫字
            self.canvas.create_text(
                x + side / 2, y + side / 2,
                anchor="center",
                text=text,
                font=font_obj,
                fill=COLOR.CAGE_SUM_FG
            )

    def flash_error(self, r, c):
        x0 = UI.BOARD_PADDING + c * UI.CELL_SIZE + UI.CELL_SIZE/2
        y0 = UI.BOARD_PADDING + r * UI.CELL_SIZE + UI.CELL_SIZE/2
        v = self.game.board[r][c].value
        self.canvas.create_text(x0, y0, text=str(v), font=FONT.FONT_BIG, fill=COLOR.ERR_FG, tag="err")
        self.canvas.after(300, lambda: self.canvas.delete("err"))

    # ---- 狀態 ----
    def update_clock(self):
        self.status.config(text=f"Mistakes: {self.game.mistakes}   Time: {self.game.elapsed_str()}")
        self.root.after(500, self.update_clock)