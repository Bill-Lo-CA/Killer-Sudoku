import tempfile
import unittest
from pathlib import Path

import torch

from generator import SudokuGenerator
from models import Cage
from rl import cage_map, is_valid_solution, load_model, solve_puzzle, step_reward, train_model, valid_digits


class KillerSudokuModelTests(unittest.TestCase):
    def test_two_cell_cage_total_17_means_8_and_9(self):
        cage = Cage(17, [(0, 0), (0, 1)])
        grid = [[0] * 9 for _ in range(9)]
        self.assertEqual(valid_digits(grid, cage_map([cage]), 0, 0), [8, 9])

    def test_rewards_include_logic_and_clear_penalty(self):
        self.assertEqual(step_reward(correct=False), -1.5)
        self.assertAlmostEqual(step_reward(correct=True, forced_moves=3), 1.3)
        self.assertEqual(step_reward(correct=True, solved=True), 11.0)

    def test_tiny_training_pipeline_solves_a_puzzle(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pth"
            results = []
            train_model(
                episodes=2,
                seed=1,
                model_path=model_path,
                device_name="cpu",
                episode_callback=lambda _model, episode, result: results.append((episode, result)),
            )
            self.assertEqual([(episode, result.puzzle_seed) for episode, result in results], [(1, 1), (2, 2)])
            self.assertGreater(sum(result.clears for _, result in results), 0)
            for _, result in results:
                self.assertEqual(result.attempts, result.correct + result.clears)
                self.assertEqual(
                    sum(kind == "clear" for *_, kind in result.trace),
                    result.clears,
                )
                for index, (row, col, _value, kind) in enumerate(result.trace):
                    if kind != "clear":
                        continue
                    correction = result.trace[index + 1]
                    self.assertEqual(correction[:2], (row, col))
                    self.assertEqual(correction[2], result.solution[row][col])
                    self.assertEqual(correction[3], "correction")

            model = load_model(model_path, torch.device("cpu"))
            cages, _ = SudokuGenerator(7).puzzle()
            trace = []
            solution, _ = solve_puzzle(cages, model, max_nodes=100_000, trace=trace)
            self.assertTrue(is_valid_solution(solution, cages))
            replay = [[0] * 9 for _ in range(9)]
            for row, col, value in trace:
                replay[row][col] = value
            self.assertEqual(replay, solution)


if __name__ == "__main__":
    unittest.main()
