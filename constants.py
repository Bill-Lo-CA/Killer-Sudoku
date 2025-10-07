from dataclasses import dataclass
from typing import Final, Tuple

# --- 視覺參數 ---
@dataclass(frozen=True)
class Ui:
    CELL_SIZE: Final[int] = 60
    GRID_SIZE: Final[int] = 9
    BOARD_PADDING: Final[int] = 20
    LINE_THIN: Final[int] = 1
    LINE_THICK: Final[int] = 3

    CAGE_LINE_WIDTH: Final[int] = 2                 # 籠子線寬（虛線）
    CAGE_LINE_SHIFT: Final[int] = 4                 # 籠子偏移
    CAGE_DASH: Final[Tuple[int, int]] = (3, 3)      # (dash_length, gap_length)
    CAGE_PAD: Final[int] = 5

@dataclass(frozen=True)
class Color:
    BG: Final[str] = "#f6eddc"
    GRID_COLOR: Final[str] = "#2f2f2f"
    NOTE_COLOR: Final[str] = "#374151"
    HILITE: Final[str] = "#cfe8ff"
    SELECT: Final[str] = "#9cc7ff"
    FIXED_FG: Final[str] = "#1d4ed8"
    USER_FG: Final[str] = "#111111"
    ERR_FG: Final[str] = "#d11"
    CAGE_SUM_FG: Final[str] = "#6b7280"

@dataclass(frozen=True)
class Font:
    FONT_BIG: Final[Tuple[str, int, str]] = ("Segoe UI", 20, "bold")
    FONT_NOTE: Final[Tuple[str, int]] = ("Segoe UI", 9)
    FONT_UI: Final[Tuple[str, int]] = ("Segoe UI", 11)

UI = Ui()
COLOR = Color()
FONT = Font()
