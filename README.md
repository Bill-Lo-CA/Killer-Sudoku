# Killer Sudoku

A small, ad-free Killer Sudoku desktop game built with Python and Tkinter.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- Tkinter (included with the standard Windows Python installer)

The game itself only uses the standard library. PyTorch is installed by uv for
the optional trained solver.

## Setup

```powershell
uv sync
```

## Run

```powershell
uv run python main.py
```

## Controls

- Click a cell, then press `1`-`9` to enter a number.
- Press `P` to toggle pencil notes.
- Press `Backspace` or `Delete` to clear the selected cell.
- Use the buttons for undo, clear, reset, and up to three hints.

## Test

```powershell
uv run python -B -m unittest -v
```

## Train and run the solver

The exact constraint layer handles cage arithmetic. For example, a two-cell
cage totalling 17 is always restricted to 8 and 9. The neural network learns
which cell should receive each candidate.

Train a model on generated puzzles:

```powershell
uv run python rl.py train --episodes 5000
```

Watch RL training on a new generated puzzle every episode:

```powershell
uv run python rl.py train --episodes 5000 --watch --delay-ms 20
```

Each episode keeps working on the same puzzle until it is solved. A contradiction
gives only the responsible candidate -1 for the dead end and -0.5 for
backtracking. A candidate whose subtree reaches a solution receives +1.5. Forced
logic is potential shaping capped at +0.25 over the complete successful path and
is cancelled by backtracking. The five-mistake challenge scores 0 through 4
mistakes as +1.0, +0.8, +0.6, +0.4, and +0.2; the fifth mistake fixes the score at
-1.0, but recovery search still completes the puzzle for branch training. Branch
loss is primary and challenge loss has weight 0.5. The generated answer is never
used for training, and choices are compared against their original legal
candidates. Console output shows mean branch reward, challenge reward, pass@5,
decisions, backtracks, dead ends, forced moves, and rolling averages for the
latest 100 solved puzzles. Training stops with an error instead of skipping a
puzzle if `--max-nodes` is exceeded.

Solve a reproducible generated puzzle:

```powershell
uv run python rl.py solve --seed 42 --max-mistakes 5
```

Watch the model's actual placements and backtracking in the Tkinter board:

```powershell
uv run python rl.py watch --seed 42 --max-mistakes 5 --delay-ms 120
```

Only failed choices between multiple candidates count as mistakes. Forced logic
moves do not. Solve and watch stop immediately when the fifth mistake is used.

The model is saved as `killer_sudoku_model.pth`, which is ignored by Git.

## Project layout

- `main.py`: application entry point
- `generator.py`: solution and cage generation
- `game.py`: game state and rules
- `ui_tk.py`: Tkinter interface
- `sudoku_full.py`: MRV and DLX Sudoku solution generators
- `rl.py`: model training and neural-guided exact solver

The trained solver is currently a command-line tool and is not connected to the
Tkinter buttons.
