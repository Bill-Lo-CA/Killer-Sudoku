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
    cells: List[Tuple[int, int]] = field(default_factory=list)
    
    def append(self, r: int, c: int) -> None:
        self.cells.append((r, c))

@dataclass
class StrictCage(Cage):
    numbers: Set[int] = field(default_factory=set)

    def append_number(self, num: int) -> bool:
        if num in self.numbers:
            return False
        else:
            self.numbers.add(num)
            return True