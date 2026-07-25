import random
import unittest

from game import Game
from generator import SudokuGenerator
from models import Cage


class KillerSudokuTests(unittest.TestCase):
    def test_seed_reproduces_complete_puzzle(self):
        self.assertEqual(SudokuGenerator(42).puzzle(), SudokuGenerator(42).puzzle())

    def test_generated_answers_satisfy_game_rules(self):
        state = random.getstate()
        try:
            random.seed(0)
            for seed in range(100):
                cages, answer = SudokuGenerator(seed).puzzle()
                game = Game({}, cages, answer)
                for r in range(9):
                    for c in range(9):
                        game.board[r][c].value = answer[r][c]
                self.assertTrue(game.is_solved(), f"seed {seed}")
        finally:
            random.setstate(state)

    def test_complete_board_must_obey_sudoku_rules(self):
        cages = [Cage(1, [(r, c)]) for r in range(9) for c in range(9)]
        game = Game({}, cages, [[1] * 9 for _ in range(9)])
        for row in game.board:
            for cell in row:
                cell.value = 1
        self.assertFalse(game.is_solved())

    def test_reset_clears_pencil_mode(self):
        game = Game({}, [], [[0] * 9 for _ in range(9)])
        game.toggle_pencil()
        game.reset()
        self.assertFalse(game.pencil_mode)

    def test_hint_can_be_undone(self):
        game = Game({}, [], [[1] * 9 for _ in range(9)])
        game.direct_hint()
        self.assertEqual((game.board[0][0].value, game.board[0][0].fixed), (1, True))
        game.pop_undo()
        self.assertEqual((game.board[0][0].value, game.board[0][0].fixed), (0, False))


if __name__ == "__main__":
    unittest.main()
