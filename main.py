import tkinter as tk
from sample_levels import get_sample_starters, get_sample_cages
from game import Game
from ui_tk import KillerSudokuApp

def main():
    starters = get_sample_starters()
    cages = get_sample_cages()

    root = tk.Tk()
    game = Game(starters=starters, cages=cages)
    KillerSudokuApp(root, game)
    root.mainloop()

if __name__ == "__main__":
    main()