import tempfile
import unittest
from pathlib import Path

import torch

from generator import SudokuGenerator
from models import Cage
from rl import (
    MistakeLimitReached,
    branch_reward,
    cage_map,
    candidate_order,
    challenge_reward,
    is_valid_solution,
    load_model,
    solve_puzzle,
    train_model,
    valid_digits,
)


class KillerSudokuModelTests(unittest.TestCase):
    def test_two_cell_cage_total_17_means_8_and_9(self):
        cage = Cage(17, [(0, 0), (0, 1)])
        grid = [[0] * 9 for _ in range(9)]
        self.assertEqual(valid_digits(grid, cage_map([cage]), 0, 0), [8, 9])

    def test_candidate_order_never_leaves_the_legal_set(self):
        scores = torch.tensor([10_000.0, 0, 0, 0, 0, 0, 0, 1_000.0, -1_000.0])
        for _ in range(100):
            order = candidate_order(scores, [8, 9])
            self.assertEqual(len(order), 2)
            self.assertCountEqual(order, [8, 9])

    def test_branch_and_challenge_reward_ratios(self):
        self.assertEqual(branch_reward(solved=False, forced_moves=50), -1.5)
        self.assertEqual(branch_reward(solved=True), 1.5)
        self.assertEqual(branch_reward(solved=True, forced_moves=81), 1.75)
        self.assertEqual(branch_reward(solved=True, forced_moves=1_000), 1.75)
        self.assertEqual(
            [challenge_reward(mistakes) for mistakes in range(5)],
            [1, 0.8, 0.6, 0.4, 0.2],
        )
        self.assertEqual(challenge_reward(5), -1)
        self.assertEqual(challenge_reward(500), -1)

    def test_solve_stops_when_the_mistake_limit_is_used(self):
        for seed in range(20):
            cages, _ = SudokuGenerator(seed).puzzle()
            _, _, mistakes = solve_puzzle(
                cages,
                max_nodes=100_000,
                max_mistakes=None,
            )
            if not mistakes:
                continue
            with self.assertRaises(MistakeLimitReached):
                solve_puzzle(cages, max_nodes=100_000, max_mistakes=1)
            return
        self.fail("Expected at least one generated puzzle to require a backtrack")

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
            self.assertEqual(
                [(episode, result.puzzle_seed) for episode, result in results],
                [(1, 1), (2, 2)],
            )
            for _, result in results:
                kinds = [kind for *_, kind in result.trace]
                self.assertNotIn("correction", kinds)
                self.assertTrue(result.solved)
                self.assertEqual(result.passed_challenge, result.backtracks < 5)
                self.assertEqual(result.challenge_score, challenge_reward(result.backtracks))
                self.assertGreaterEqual(result.backtracks, result.dead_ends)
                self.assertTrue(is_valid_solution(result.solution, result.cages))
                replay = [[0] * 9 for _ in range(9)]
                for row, col, value, _kind in result.trace:
                    replay[row][col] = value
                self.assertEqual(replay, result.solution)

            model = load_model(model_path, torch.device("cpu"))
            cages, _ = SudokuGenerator(7).puzzle()
            trace = []
            solution, _, _ = solve_puzzle(
                cages,
                model,
                max_nodes=100_000,
                max_mistakes=None,
                trace=trace,
            )
            self.assertTrue(is_valid_solution(solution, cages))
            replay = [[0] * 9 for _ in range(9)]
            for row, col, value in trace:
                replay[row][col] = value
            self.assertEqual(replay, solution)


if __name__ == "__main__":
    unittest.main()
