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

The window title labels model choices, deterministic logic, clears, and
environment corrections. Rewards are +1 for a correct first choice, +0.1 per
forced logic move, -1 for a wrong choice, -0.5 for clearing it, and +10 when a
correct model choice solves the puzzle. After a mistake, the environment fills
the known answer without counting it as a correct model choice. The answer is
never part of the model input. Console output includes the current episode and
rolling averages for the latest 100 episodes.

Solve a reproducible generated puzzle:

```powershell
uv run python rl.py solve --seed 42
```

Watch the model's actual placements and backtracking in the Tkinter board:

```powershell
uv run python rl.py watch --seed 42 --delay-ms 120
```

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
