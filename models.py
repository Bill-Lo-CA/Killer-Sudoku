from dataclasses import dataclass, field
from typing import List, Tuple, Set

@dataclass
class Cell:
    value: int = 0
    fixed: bool = False
    notes: Set[int] = field(default_factory=set)

@dataclass
class Cage:
    total: int
    cells: List[Tuple[int, int]]  # list of (r, c)