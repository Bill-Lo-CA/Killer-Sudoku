# sudoku_full_random.py
# 產生「完整且合法」的 9x9 數獨解，提供兩種後端：
# 1) MRV + 位元遮罩（回溯，解空間隨機化）
# 2) Algorithm X（Dancing Links, DLX），隨機打破平手與遍歷順序
#
# 用法：
#   python sudoku_full_random.py [mrv|dlx] [seed]
#
# 也可 import：
#   from sudoku_full_random import generate_full_solution
#   board = generate_full_solution(backend="dlx", seed=42)

from __future__ import annotations
import random
import sys
from typing import List, Optional, Tuple

GRID = 9
BOX = 3
DIGITS = range(1, 10)
FULL_MASK = (1 << 9) - 1  # 0b111111111

Board = List[List[int]]

# ---------------------------------------
# 共用：小工具
# ---------------------------------------

def bit(d: int) -> int:
    return 1 << (d - 1)

def box_id(r: int, c: int) -> int:
    return (r // BOX) * BOX + (c // BOX)

def mask_to_values(m: int) -> List[int]:
    vals, k = [], 1
    while m:
        if m & 1:
            vals.append(k)
        m >>= 1
        k += 1
    return vals

def blank_board() -> Board:
    return [[0] * GRID for _ in range(GRID)]

def format_board(bd: Board) -> str:
    lines = []
    for r in range(GRID):
        if r and r % 3 == 0:
            lines.append("------+-------+------")
        row = []
        for c in range(GRID):
            if c and c % 3 == 0:
                row.append("|")
            row.append(str(bd[r][c]))
        lines.append(" ".join(row))
    return "\n".join(lines)

# ---------------------------------------
# 後端 A：MRV + 位元遮罩（隨機化）
# ---------------------------------------

def generate_full_solution_mrv(seed: Optional[int] = None) -> Board:
    rng = random.Random(seed)
    bd = blank_board()
    rows = [0] * GRID
    cols = [0] * GRID
    boxes = [0] * GRID

    # 先隨機填 3 個對角 3x3 宮（可大幅減少回溯深度）
    for d in range(0, GRID, BOX):
        digits = list(DIGITS)
        rng.shuffle(digits)
        k = 0
        for r in range(d, d + BOX):
            for c in range(d, d + BOX):
                v = digits[k]; k += 1
                m = bit(v)
                bd[r][c] = v
                rows[r] |= m
                cols[c] |= m
                boxes[box_id(r, c)] |= m

    def candidates_mask(r: int, c: int) -> int:
        return (~(rows[r] | cols[c] | boxes[box_id(r, c)])) & FULL_MASK

    def choose_cell() -> Optional[Tuple[int, int, List[int]]]:
        # MRV：找候選數最少的空格；如有平手，隨機挑一格
        best_count = 10
        best_cells = []  # (r,c,vals)
        for r in range(GRID):
            for c in range(GRID):
                if bd[r][c] != 0:
                    continue
                m = candidates_mask(r, c)
                cnt = m.bit_count()
                if cnt == 0:
                    return (r, c, [])  # 死路
                if cnt < best_count:
                    best_count = cnt
                    best_cells = [(r, c, mask_to_values(m))]
                elif cnt == best_count:
                    best_cells.append((r, c, mask_to_values(m)))
        if not best_cells:
            return None
        r, c, vals = rng.choice(best_cells)
        rng.shuffle(vals)
        return (r, c, vals)

    def dfs() -> bool:
        nxt = choose_cell()
        if nxt is None:
            return True  # 全部填完
        r, c, vals = nxt
        if not vals:
            return False
        b = box_id(r, c)
        for v in vals:
            m = bit(v)
            bd[r][c] = v
            rows[r] |= m
            cols[c] |= m
            boxes[b] |= m
            if dfs():
                return True
            # 回溯
            bd[r][c] = 0
            rows[r] ^= m
            cols[c] ^= m
            boxes[b] ^= m
        return False

    ok = dfs()
    if not ok:
        # 理論上不太會發生；保險起見重試（不同 seed）
        return generate_full_solution_mrv(None)
    return bd

# ---------------------------------------
# 後端 B：Algorithm X（Dancing Links, DLX）
# ---------------------------------------

# DLX 節點與欄頭
class DLXNode:
    __slots__ = ("L", "R", "U", "D", "C", "row_id")
    def __init__(self):
        self.L: "DLXNode" = self
        self.R: "DLXNode" = self
        self.U: "DLXNode" = self
        self.D: "DLXNode" = self
        self.C: "DLXColumn" = None  # type: ignore
        self.row_id: int = -1       # 對應 (r,c,d) 的索引

class DLXColumn(DLXNode):
    __slots__ = ("size", "name")
    def __init__(self, name: int):
        super().__init__()
        self.size: int = 0
        self.name: int = name  # 可用來識別欄位

class DLX:
    def __init__(self, num_cols: int, rng: random.Random):
        self.rng = rng
        self.root = DLXColumn(-1)
        # 建立所有欄頭並雙向串聯
        prev = self.root
        self.cols: List[DLXColumn] = []
        for i in range(num_cols):
            col = DLXColumn(i)
            self.cols.append(col)
            # 橫向串起來
            col.R = prev.R
            col.L = prev
            prev.R.L = col
            prev.R = col
            # 縱向初始化
            col.U = col.D = col
            prev = col
        self.solution_rows: List[DLXNode] = []

    def cover(self, c: DLXColumn):
        # 從 header 列移除欄 c
        c.R.L = c.L
        c.L.R = c.R
        # 並移除此欄的所有列（該列的其他欄也要更新 size）
        i = c.D
        while i is not c:
            j = i.R
            while j is not i:
                j.D.U = j.U
                j.U.D = j.D
                j.C.size -= 1
                j = j.R
            i = i.D

    def uncover(self, c: DLXColumn):
        # 還原 cover 的操作（逆序）
        i = c.U
        while i is not c:
            j = i.L
            while j is not i:
                j.C.size += 1
                j.D.U = j
                j.U.D = j
                j = j.L
            i = i.U
        c.R.L = c
        c.L.R = c

    def choose_column(self) -> DLXColumn:
        # S heuristic：選取 size 最小的欄；平手時隨機
        c = self.root.R  # type: ignore
        min_size = 10**9
        candidates: List[DLXColumn] = []
        while isinstance(c, DLXColumn) and c is not self.root:
            if c.size < min_size:
                min_size = c.size
                candidates = [c]
            elif c.size == min_size:
                candidates.append(c)
            c = c.R  # type: ignore
        return self.rng.choice(candidates)

    def search(self) -> bool:
        # 若 header 只剩 root，代表所有約束都滿足
        if self.root.R is self.root:
            return True
        c = self.choose_column()
        if c.size == 0:
            return False
        self.cover(c)

        # 走訪此欄的所有節點（各代表一個選項），並打亂順序增加隨機性
        rows: List[DLXNode] = []
        r = c.D
        while r is not c:
            rows.append(r)
            r = r.D
        self.rng.shuffle(rows)

        for r in rows:
            self.solution_rows.append(r)
            # cover 該列的其他欄
            j = r.R
            while j is not r:
                self.cover(j.C)
                j = j.R
            # 遞迴
            if self.search():
                return True
            # 回溯
            j = r.L
            while j is not r:
                self.uncover(j.C)
                j = j.L
            self.solution_rows.pop()
        self.uncover(c)
        return False

# 建立 9x9 Sudoku 的精確覆蓋矩陣（不實作成大矩陣，而是直接建 DLX 結構）
# 共有 324 個欄位（4*81 約束），729 個選項列（r,c,d）
# 欄位編碼：
# 0..80: cell(r,c)
# 81..161: row(r,d)
# 162..242: col(c,d)
# 243..323: box(b,d), b = (r//3)*3 + (c//3)
def build_dlx_for_sudoku(rng: random.Random) -> DLX:
    NUM_COLS = 324
    dlx = DLX(NUM_COLS, rng)

    def append_row(col_indices: List[int], row_id: int):
        first: Optional[DLXNode] = None
        prev: Optional[DLXNode] = None
        for ci in col_indices:
            col = dlx.cols[ci]
            node = DLXNode()
            node.C = col
            node.row_id = row_id
            # 垂直插入（在 col.U 和 col 之間）
            node.U = col.U
            node.D = col
            col.U.D = node
            col.U = node
            col.size += 1
            # 水平串聯
            if first is None:
                first = node
            if prev is not None:
                node.L = prev
                node.R = prev.R
                prev.R.L = node
                prev.R = node
            prev = node
        # 完成一圈
        if first is not None and prev is not None:
            first.L = prev
            prev.R = first

    # 建所有 729 列（可先打亂 (r,c,d) 的遍歷順序以增加隨機）
    triples = [(r, c, d) for r in range(GRID) for c in range(GRID) for d in DIGITS]
    rng.shuffle(triples)
    for r, c, d in triples:
        cell = r * 9 + c
        row_d = 81 + r * 9 + (d - 1)
        col_d = 162 + c * 9 + (d - 1)
        box_d = 243 + ((r // 3) * 3 + (c // 3)) * 9 + (d - 1)
        append_row([cell, row_d, col_d, box_d], row_id=r * 81 + c * 9 + (d - 1))
    return dlx

def generate_full_solution_dlx(seed: Optional[int] = None) -> Board:
    rng = random.Random(seed)
    dlx = build_dlx_for_sudoku(rng)
    ok = dlx.search()
    if not ok:
        # 幾乎不會；保險起見
        return generate_full_solution_dlx(None)

    # 將解轉為 9x9
    bd = blank_board()
    for node in dlx.solution_rows:
        rid = node.row_id  # r*81 + c*9 + (d-1)
        r = rid // 81
        c = (rid // 9) % 9
        d = (rid % 9) + 1
        bd[r][c] = d
    return bd

# ---------------------------------------
# API
# ---------------------------------------

def generate_full_solution(backend: str = "mrv", seed: Optional[int] = None) -> Board:
    backend = backend.lower()
    if backend == "dlx":
        return generate_full_solution_dlx(seed)
    # 預設 MRV
    return generate_full_solution_mrv(seed)

# ---------------------------------------
# CLI
# ---------------------------------------

if __name__ == "__main__":
    backend = "mrv"
    seed: Optional[int] = None
    if len(sys.argv) >= 2:
        backend = sys.argv[1]
    if len(sys.argv) >= 3:
        try:
            seed = int(sys.argv[2])
        except ValueError:
            seed = None

    board = generate_full_solution(backend=backend, seed=seed)
    print(f"# backend={backend}, seed={seed}")
    print(format_board(board))