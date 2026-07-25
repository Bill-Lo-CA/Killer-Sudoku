# Killer Sudoku

A small, ad-free Killer Sudoku desktop game built with Python and Tkinter.

## Requirements

- Python 3.10 or newer
- Tkinter (included with the standard Windows Python installer)

The game itself has no third-party dependencies.

## Run

```powershell
python main.py
```

## Controls

- Click a cell, then press `1`-`9` to enter a number.
- Press `P` to toggle pencil notes.
- Press `Backspace` or `Delete` to clear the selected cell.
- Use the buttons for undo, clear, reset, and up to three hints.

## Test

```powershell
python -B -m unittest -v
```

## Project layout

- `main.py`: application entry point
- `generator.py`: solution and cage generation
- `game.py`: game state and rules
- `ui_tk.py`: Tkinter interface
- `sudoku_full.py`: MRV and DLX Sudoku solution generators

`rl.py` is an experimental reinforcement-learning prototype. It is not used by
the game entry point and requires NumPy and PyTorch if developed further.
