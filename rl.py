from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn

from generator import SudokuGenerator
from models import Cage
from sudoku_full import format_board


SIZE = 9
DIGITS = tuple(range(1, 10))
CHANNELS = 16  # grid, cage sum/size/borders, and one candidate mask per digit
BRANCH_SUCCESS_REWARD = 1.5
BRANCH_FAILURE_REWARD = -1.0
BACKTRACK_REWARD = -0.5
FORCED_SHAPING_MAX = 0.25
CHALLENGE_WEIGHT = 0.5
MISTAKE_BUDGET = 5


class MistakeLimitReached(RuntimeError):
    pass


@lru_cache(maxsize=None)
def cage_combinations(size: int, total: int) -> tuple[tuple[int, ...], ...]:
    """All distinct digit combinations allowed by one cage."""
    return tuple(combo for combo in combinations(DIGITS, size) if sum(combo) == total)


def cage_map(cages: Iterable[Cage]) -> dict[tuple[int, int], Cage]:
    return {cell: cage for cage in cages for cell in cage.cells}


def valid_digits(
    grid: list[list[int]],
    cages_by_cell: dict[tuple[int, int], Cage],
    row: int,
    col: int,
) -> list[int]:
    """Return digits allowed by Sudoku and exact cage-combination rules."""
    if grid[row][col]:
        return []

    used = set(grid[row])
    used.update(grid[r][col] for r in range(SIZE))
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    used.update(
        grid[r][c]
        for r in range(box_row, box_row + 3)
        for c in range(box_col, box_col + 3)
    )
    used.discard(0)

    cage = cages_by_cell.get((row, col))
    if cage is None:
        return [digit for digit in DIGITS if digit not in used]

    placed = [grid[r][c] for r, c in cage.cells if grid[r][c]]
    if len(placed) != len(set(placed)):
        return []

    placed_set = set(placed)
    possible = {
        digit
        for combo in cage_combinations(len(cage), cage.total)
        if placed_set.issubset(combo)
        for digit in combo
        if digit not in placed_set
    }
    return sorted(possible - used)


def encode_state(cages: list[Cage], grid: list[list[int]]) -> torch.Tensor:
    """Encode a puzzle as CNN channels, including exact per-digit candidates."""
    state = torch.zeros((CHANNELS, SIZE, SIZE), dtype=torch.float32)
    state[0] = torch.tensor(grid, dtype=torch.float32) / 9.0
    by_cell = cage_map(cages)

    for cage in cages:
        cells = set(cage.cells)
        for row, col in cells:
            state[1, row, col] = cage.total / 45.0
            state[2, row, col] = len(cage) / 9.0
            state[3, row, col] = (row - 1, col) not in cells
            state[4, row, col] = (row, col + 1) not in cells
            state[5, row, col] = (row + 1, col) not in cells
            state[6, row, col] = (row, col - 1) not in cells

    for row in range(SIZE):
        for col in range(SIZE):
            for digit in valid_digits(grid, by_cell, row, col):
                state[7 + digit - 1, row, col] = 1.0

    return state


class KillerSudokuNet(nn.Module):
    """Small CNN that scores each digit for every cell."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(CHANNELS, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 9, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


def choose_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def save_model(model: KillerSudokuNet, path: str | Path) -> None:
    torch.save(
        {
            "channels": CHANNELS,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        },
        path,
    )


def load_model(path: str | Path, device: torch.device) -> KillerSudokuNet:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("channels") != CHANNELS:
        raise ValueError("Model uses an incompatible puzzle encoding")
    model = KillerSudokuNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def branch_reward(*, solved: bool, forced_moves: int = 0) -> float:
    if not solved:
        return BRANCH_FAILURE_REWARD + BACKTRACK_REWARD
    forced = min(max(forced_moves, 0), SIZE * SIZE)
    return BRANCH_SUCCESS_REWARD + FORCED_SHAPING_MAX * forced / (SIZE * SIZE)


def challenge_reward(mistakes: int) -> float:
    if mistakes < 0:
        raise ValueError("mistakes cannot be negative")
    if mistakes >= MISTAKE_BUDGET:
        return -1.0
    return (MISTAKE_BUDGET - mistakes) / MISTAKE_BUDGET


def candidate_order(scores: torch.Tensor, candidates: list[int]) -> list[int]:
    """Sample a weighted permutation using only legal candidate logits."""
    indices = torch.tensor([digit - 1 for digit in candidates], device=scores.device)
    candidate_scores = scores[indices]
    noise = torch.empty_like(candidate_scores).exponential_()
    noise.clamp_min_(torch.finfo(noise.dtype).tiny)
    ranking = torch.argsort(candidate_scores - noise.log(), descending=True)
    return [candidates[index] for index in ranking.tolist()]


def apply_forced_moves(
    cages: list[Cage],
    grid: list[list[int]],
    trace: list[tuple[int, int, int, str]] | None = None,
) -> tuple[int, bool]:
    """Fill singleton candidates and report whether the resulting state is valid."""
    by_cell = cage_map(cages)
    filled = 0
    while True:
        moved = False
        for row in range(SIZE):
            for col in range(SIZE):
                if grid[row][col]:
                    continue
                candidates = valid_digits(grid, by_cell, row, col)
                if not candidates:
                    return filled, False
                if len(candidates) == 1:
                    grid[row][col] = candidates[0]
                    if trace is not None:
                        trace.append((row, col, candidates[0], "logic"))
                    filled += 1
                    moved = True
                    break
            if moved:
                break
        if not moved:
            return filled, True


def choose_branch(
    grid: list[list[int]],
    by_cell: dict[tuple[int, int], Cage],
) -> tuple[int, int, list[int]] | None:
    best: tuple[int, int, list[int]] | None = None
    for row in range(SIZE):
        for col in range(SIZE):
            if grid[row][col]:
                continue
            candidates = valid_digits(grid, by_cell, row, col)
            if not candidates:
                raise ValueError("Training state has no legal action")
            if len(candidates) > 1 and (best is None or len(candidates) < len(best[2])):
                best = row, col, candidates
    return best


@dataclass
class EpisodeResult:
    puzzle_seed: int
    cages: list[Cage]
    solution: list[list[int]]
    trace: list[tuple[int, int, int, str]]
    reward: float
    challenge_score: float
    passed_challenge: bool
    attempts: int
    backtracks: int
    dead_ends: int
    forced_moves: int
    solved: bool


@dataclass
class Decision:
    state: torch.Tensor
    row: int
    col: int
    candidates: tuple[int, ...]
    digit: int


def decision_policy(
    model: KillerSudokuNet,
    decisions: list[Decision],
    *,
    device: torch.device,
) -> torch.distributions.Categorical:
    states = torch.stack([decision.state for decision in decisions]).to(device)
    output = model(states)
    scores = torch.stack(
        [output[index, :, decision.row, decision.col] for index, decision in enumerate(decisions)]
    )
    masked_scores = torch.full_like(scores, float("-inf"))
    for index, decision in enumerate(decisions):
        indices = [digit - 1 for digit in decision.candidates]
        masked_scores[index, indices] = scores[index, indices]
    return torch.distributions.Categorical(logits=masked_scores)


def path_loss(
    model: KillerSudokuNet,
    decisions: list[Decision],
    reward: float,
    *,
    device: torch.device,
    gamma: float,
) -> torch.Tensor:
    policy = decision_policy(model, decisions, device=device)
    actions = torch.tensor([decision.digit - 1 for decision in decisions], device=device)
    returns = torch.tensor(
        [reward * gamma ** (len(decisions) - index - 1) for index in range(len(decisions))],
        device=device,
    )
    return (-policy.log_prob(actions) * returns).mean()


def branch_loss(
    model: KillerSudokuNet,
    samples: list[tuple[Decision, float]],
    *,
    device: torch.device,
    entropy_weight: float,
) -> torch.Tensor:
    decisions = [decision for decision, _reward in samples]
    rewards = torch.tensor([reward for _decision, reward in samples], device=device)
    policy = decision_policy(model, decisions, device=device)
    actions = torch.tensor([decision.digit - 1 for decision in decisions], device=device)
    losses = -policy.log_prob(actions) * rewards
    groups = []
    for successful in (False, True):
        group = losses[(rewards > 0) == successful]
        if group.numel():
            groups.append(group.mean())
    return torch.stack(groups).mean() - entropy_weight * policy.entropy().mean()


def train_policy(
    model: KillerSudokuNet,
    optimizer: torch.optim.Optimizer,
    branch_samples: list[tuple[Decision, float]],
    challenge_path: tuple[list[Decision], float] | None,
    *,
    device: torch.device,
    entropy_weight: float,
    gamma: float,
) -> None:
    losses = []
    model.train()
    if branch_samples:
        losses.append(
            branch_loss(
                model,
                branch_samples,
                device=device,
                entropy_weight=entropy_weight,
            )
        )
    if challenge_path is not None and challenge_path[0]:
        decisions, reward = challenge_path
        losses.append(
            CHALLENGE_WEIGHT
            * path_loss(
                model,
                decisions,
                reward,
                device=device,
                gamma=gamma,
            )
        )
    if not losses:
        return
    optimizer.zero_grad()
    sum(losses).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def train_episode(
    model: KillerSudokuNet,
    optimizer: torch.optim.Optimizer,
    *,
    puzzle_seed: int,
    device: torch.device,
    entropy_weight: float,
    gamma: float,
    max_nodes: int,
    collect_trace: bool,
) -> EpisodeResult:
    cages, _generated_solution = SudokuGenerator(puzzle_seed).puzzle()
    grid = [[0] * SIZE for _ in range(SIZE)]
    by_cell = cage_map(cages)
    trace: list[tuple[int, int, int, str]] = []
    active_path: list[Decision] = []
    branch_rewards: list[float] = []
    branch_samples: deque[tuple[Decision, float]] = deque(maxlen=256)
    challenge_mistakes: list[Decision] = []
    attempts = backtracks = dead_ends = forced_total = 0

    initial_forced, valid = apply_forced_moves(cages, grid, trace if collect_trace else None)
    forced_total += initial_forced
    if not valid:
        raise RuntimeError("Generated puzzle starts in a contradictory state")

    def restore(snapshot: list[list[int]]) -> None:
        for row in range(SIZE):
            for col in range(SIZE):
                if grid[row][col] == snapshot[row][col]:
                    continue
                grid[row][col] = snapshot[row][col]
                if collect_trace:
                    trace.append((row, col, snapshot[row][col], "clear"))

    def search() -> bool:
        nonlocal attempts, backtracks, dead_ends, forced_total
        branch = choose_branch(grid, by_cell)
        if branch is None:
            solved = is_valid_solution(grid, cages)
            if not solved:
                dead_ends += 1
            return solved

        row, col, candidates = branch
        state = encode_state(cages, grid)
        model.eval()
        with torch.no_grad():
            scores = model(state.unsqueeze(0).to(device))[0, :, row, col]
            order = candidate_order(scores, candidates)

        for digit in order:
            attempts += 1
            if attempts > max_nodes:
                raise RuntimeError(f"Training search exceeded {max_nodes} nodes")

            snapshot = [values[:] for values in grid]
            decision = Decision(state, row, col, tuple(candidates), digit)
            active_path.append(decision)
            grid[row][col] = digit
            if collect_trace:
                trace.append((row, col, digit, "model"))

            forced, branch_valid = apply_forced_moves(
                cages,
                grid,
                trace if collect_trace else None,
            )
            forced_total += forced
            solved = False
            if branch_valid:
                solved = search()
            else:
                dead_ends += 1

            if solved:
                reward = branch_reward(solved=True, forced_moves=forced)
                branch_rewards.append(reward)
                branch_samples.append((decision, reward))
                return True

            reward = branch_reward(solved=False)
            branch_rewards.append(reward)
            branch_samples.append((decision, reward))
            active_path.pop()
            restore(snapshot)
            backtracks += 1
            if len(challenge_mistakes) < MISTAKE_BUDGET:
                challenge_mistakes.append(decision)

        return False

    solved = is_valid_solution(grid, cages) if all(all(row) for row in grid) else search()
    if not solved:
        raise RuntimeError("Generated puzzle could not be solved during training")

    passed_challenge = backtracks < MISTAKE_BUDGET
    score = challenge_reward(backtracks)
    challenge_path = (
        (active_path.copy(), score)
        if passed_challenge
        else (challenge_mistakes, score)
    )
    train_policy(
        model,
        optimizer,
        list(branch_samples),
        challenge_path,
        device=device,
        entropy_weight=entropy_weight,
        gamma=gamma,
    )

    solution = [row[:] for row in grid]
    reward = sum(branch_rewards) / len(branch_rewards) if branch_rewards else 0.0

    return EpisodeResult(
        puzzle_seed=puzzle_seed,
        cages=cages,
        solution=solution,
        trace=trace,
        reward=reward,
        challenge_score=score,
        passed_challenge=passed_challenge,
        attempts=attempts,
        backtracks=backtracks,
        dead_ends=dead_ends,
        forced_moves=forced_total,
        solved=True,
    )


def train_model(
    *,
    episodes: int = 1_000,
    learning_rate: float = 1e-3,
    entropy_weight: float = 0.01,
    gamma: float = 0.99,
    max_nodes: int = 1_000_000,
    seed: int = 0,
    model_path: str | Path = "killer_sudoku_model.pth",
    device_name: str = "auto",
    episode_callback: Callable[[KillerSudokuNet, int, EpisodeResult], None] | None = None,
) -> KillerSudokuNet:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")

    torch.manual_seed(seed)
    device = choose_device(device_name)
    model = KillerSudokuNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    recent_rewards: deque[float] = deque(maxlen=100)
    recent_challenge_scores: deque[float] = deque(maxlen=100)
    recent_decisions: deque[int] = deque(maxlen=100)
    recent_backtracks: deque[int] = deque(maxlen=100)
    recent_within_five: deque[bool] = deque(maxlen=100)

    print(f"Training on {device}")
    for episode in range(1, episodes + 1):
        result = train_episode(
            model,
            optimizer,
            puzzle_seed=seed + episode - 1,
            device=device,
            entropy_weight=entropy_weight,
            gamma=gamma,
            max_nodes=max_nodes,
            collect_trace=episode_callback is not None,
        )
        recent_rewards.append(result.reward)
        recent_challenge_scores.append(result.challenge_score)
        recent_decisions.append(result.attempts)
        recent_backtracks.append(result.backtracks)
        recent_within_five.append(result.passed_challenge)
        if episode == 1 or episode % 10 == 0 or episode == episodes:
            window = len(recent_rewards)
            print(
                f"Episode {episode}/{episodes} - mean branch reward: {result.reward:.2f} - "
                f"challenge: {result.challenge_score:+.1f} - "
                f"decisions: {result.attempts} - backtracks: {result.backtracks} - "
                f"dead ends: {result.dead_ends} - forced: {result.forced_moves} | "
                f"avg{window} branch reward: {sum(recent_rewards) / window:.2f} - "
                f"challenge: {sum(recent_challenge_scores) / window:+.2f} - "
                f"decisions: {sum(recent_decisions) / window:.1f} - backtracks: "
                f"{sum(recent_backtracks) / window:.1f} - pass@5: "
                f"{sum(recent_within_five) / window:.1%}"
            )
        if episode_callback is not None:
            episode_callback(model, episode, result)

    save_model(model, model_path)
    print(f"Saved model to {model_path}")
    return model


def is_valid_solution(grid: list[list[int]], cages: list[Cage]) -> bool:
    expected = set(DIGITS)
    if any(set(row) != expected for row in grid):
        return False
    if any({grid[row][col] for row in range(SIZE)} != expected for col in range(SIZE)):
        return False
    for box_row in range(0, SIZE, 3):
        for box_col in range(0, SIZE, 3):
            values = {
                grid[row][col]
                for row in range(box_row, box_row + 3)
                for col in range(box_col, box_col + 3)
            }
            if values != expected:
                return False
    return all(
        sum(grid[row][col] for row, col in cage.cells) == cage.total
        and len({grid[row][col] for row, col in cage.cells}) == len(cage)
        for cage in cages
    )


def solve_puzzle(
    cages: list[Cage],
    model: KillerSudokuNet | None = None,
    *,
    device: torch.device | None = None,
    max_nodes: int = 1_000_000,
    max_mistakes: int | None = MISTAKE_BUDGET,
    trace: list[tuple[int, int, int]] | None = None,
) -> tuple[list[list[int]], int, int]:
    """Solve with exact constraints; use the model only to order candidates."""
    if max_mistakes is not None and max_mistakes < 1:
        raise ValueError("max_mistakes must be positive or None")

    grid = [[0] * SIZE for _ in range(SIZE)]
    by_cell = cage_map(cages)
    device = device or torch.device("cpu")
    nodes = 0
    mistakes = 0

    if model is not None:
        model.eval()

    def search() -> bool:
        nonlocal mistakes, nodes
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError(f"Search exceeded {max_nodes} nodes")

        best: tuple[int, int, list[int]] | None = None
        for row in range(SIZE):
            for col in range(SIZE):
                if grid[row][col]:
                    continue
                candidates = valid_digits(grid, by_cell, row, col)
                if not candidates:
                    return False
                if best is None or len(candidates) < len(best[2]):
                    best = row, col, candidates
                    if len(candidates) == 1:
                        break
            if best is not None and len(best[2]) == 1:
                break

        if best is None:
            return is_valid_solution(grid, cages)

        row, col, candidates = best
        if model is not None and len(candidates) > 1:
            with torch.no_grad():
                state = encode_state(cages, grid).unsqueeze(0).to(device)
                scores = model(state)[0, :, row, col]
            candidates.sort(key=lambda digit: scores[digit - 1].item(), reverse=True)

        for digit in candidates:
            grid[row][col] = digit
            if trace is not None:
                trace.append((row, col, digit))
            if search():
                return True
            grid[row][col] = 0
            if trace is not None:
                trace.append((row, col, 0))
            if len(candidates) > 1:
                mistakes += 1
                if max_mistakes is not None and mistakes >= max_mistakes:
                    raise MistakeLimitReached(
                        f"used all {max_mistakes} allowed mistakes after {nodes} search nodes"
                    )
        return False

    if not search():
        raise ValueError("Puzzle has no solution")
    return grid, nodes, mistakes


def watch_solution(
    cages: list[Cage],
    solution: list[list[int]],
    trace: list[tuple[int, int, int]],
    nodes: int,
    mistakes: int,
    max_mistakes: int,
    delay_ms: int,
) -> None:
    if delay_ms < 1:
        raise ValueError("delay_ms must be positive")

    import tkinter as tk

    from game import Game
    from ui_tk import KillerSudokuApp

    root = tk.Tk()
    game = Game(starters={}, cages=cages, ans=solution)
    ui = KillerSudokuApp(root, game)
    step = 0

    def play_next() -> None:
        nonlocal step
        if step == len(trace):
            root.title(
                f"Killer Sudoku AI - solved in {nodes} search nodes - "
                f"mistakes {mistakes}/{max_mistakes}"
            )
            return

        row, col, value = trace[step]
        game.select(row, col)
        cell = game.board[row][col]
        cell.value = value
        cell.notes.clear()
        step += 1
        root.title(f"Killer Sudoku AI - step {step}/{len(trace)}")
        ui.redraw()
        root.after(delay_ms, play_next)

    root.after(delay_ms, play_next)
    root.mainloop()


def watch_training(args: argparse.Namespace) -> None:
    import queue
    import threading
    import tkinter as tk
    import traceback
    from tkinter import messagebox

    from game import Game
    from ui_tk import KillerSudokuApp

    if args.delay_ms < 1:
        raise ValueError("delay_ms must be positive")

    messages: queue.Queue = queue.Queue()
    closed = threading.Event()
    current_gate: threading.Event | None = None

    root = tk.Tk()
    cages, expected = SudokuGenerator(args.seed).puzzle()
    game = Game(starters={}, cages=cages, ans=expected)
    ui = KillerSudokuApp(root, game)
    root.title("Killer Sudoku AI - preparing training")

    def on_episode(_model: KillerSudokuNet, episode: int, result: EpisodeResult) -> None:
        if closed.is_set():
            return
        gate = threading.Event()
        messages.put(("episode", episode, result, gate))
        while not closed.is_set() and not gate.wait(0.1):
            pass

    def train_worker() -> None:
        try:
            train_model(
                episodes=args.episodes,
                learning_rate=args.learning_rate,
                entropy_weight=args.entropy_weight,
                gamma=args.gamma,
                max_nodes=args.max_nodes,
                seed=args.seed,
                model_path=args.model,
                device_name=args.device,
                episode_callback=on_episode,
            )
            messages.put(("done",))
        except Exception:
            messages.put(("error", traceback.format_exc()))

    def play_episode(message: tuple) -> None:
        nonlocal current_gate
        _, episode, result, gate = message
        current_gate = gate
        game.cages = result.cages
        game.ans = result.solution
        game.reset()
        ui.cage_data = game.cage_data()
        step = 0

        def play_next() -> None:
            nonlocal step, current_gate
            if closed.is_set():
                gate.set()
                return
            if step == len(result.trace):
                root.title(
                    f"Episode {episode}/{args.episodes} - mean branch reward "
                    f"{result.reward:.2f} - challenge {result.challenge_score:+.1f} - "
                    f"solved - decisions {result.attempts} - backtracks "
                    f"{result.backtracks} - forced {result.forced_moves}"
                )
                current_gate = None
                gate.set()
                root.after(50, poll_messages)
                return

            row, col, value, kind = result.trace[step]
            game.select(row, col)
            game.board[row][col].value = value
            step += 1
            root.title(
                f"Episode {episode}/{args.episodes} - {kind} - "
                f"step {step}/{len(result.trace)} - reward {result.reward:.1f}"
            )
            ui.redraw()
            root.after(args.delay_ms, play_next)

        root.after(args.delay_ms, play_next)

    def poll_messages() -> None:
        if closed.is_set():
            return
        try:
            message = messages.get_nowait()
        except queue.Empty:
            root.after(50, poll_messages)
            return

        if message[0] == "episode":
            play_episode(message)
        elif message[0] == "done":
            root.title(f"Training complete - saved {args.model}")
        else:
            root.title("Training failed")
            messagebox.showerror("Training failed", message[1])

    def close() -> None:
        closed.set()
        if current_gate is not None:
            current_gate.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    threading.Thread(target=train_worker, daemon=True).start()
    root.after(50, poll_messages)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run the Killer Sudoku model")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a model on generated puzzles")
    train.add_argument("--episodes", type=int, default=1_000)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--entropy-weight", type=float, default=0.01)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--max-nodes", type=int, default=1_000_000)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--model", default="killer_sudoku_model.pth")
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--watch", action="store_true", help="visualize each training episode")
    train.add_argument("--delay-ms", type=int, default=20)

    solve = commands.add_parser("solve", help="solve a generated puzzle with a trained model")
    watch = commands.add_parser("watch", help="watch the real search trace in Tkinter")
    for command in (solve, watch):
        command.add_argument("--model", default="killer_sudoku_model.pth")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--max-nodes", type=int, default=1_000_000)
        command.add_argument("--max-mistakes", type=int, default=MISTAKE_BUDGET)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    watch.add_argument("--delay-ms", type=int, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        if args.watch:
            watch_training(args)
            return
        train_model(
            episodes=args.episodes,
            learning_rate=args.learning_rate,
            entropy_weight=args.entropy_weight,
            gamma=args.gamma,
            max_nodes=args.max_nodes,
            seed=args.seed,
            model_path=args.model,
            device_name=args.device,
        )
        return

    device = choose_device(args.device)
    model = load_model(args.model, device)
    cages, _ = SudokuGenerator(args.seed).puzzle()
    trace = [] if args.command == "watch" else None
    try:
        solution, nodes, mistakes = solve_puzzle(
            cages,
            model,
            device=device,
            max_nodes=args.max_nodes,
            max_mistakes=args.max_mistakes,
            trace=trace,
        )
    except MistakeLimitReached as error:
        raise SystemExit(f"Failed seed {args.seed}: {error}") from None
    if trace is not None:
        watch_solution(
            cages,
            solution,
            trace,
            nodes,
            mistakes,
            args.max_mistakes,
            args.delay_ms,
        )
        return
    print(
        f"Solved seed {args.seed} in {nodes} search nodes with "
        f"{mistakes}/{args.max_mistakes} mistakes on {device}"
    )
    print(format_board(solution))


if __name__ == "__main__":
    main()
