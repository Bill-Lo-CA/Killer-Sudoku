from dataclasses import dataclass, field
from typing import List, Tuple, Set

@dataclass
class Cell:
    value: int = 0
    fixed: bool = False
    notes: Set[int] = field(default_factory=set)

@dataclass
class Cage:
    total: int = field(default_factory=int)
    cells: List[Tuple[int, int]] = field(default_factory=list)
    
    def append(self, r: int, c: int) -> None:
        self.cells.append((r, c))
    
    def add_total(self, num: int) -> None:
        self.total += num

    def __len__(self) -> int:
        return len(self.cells)