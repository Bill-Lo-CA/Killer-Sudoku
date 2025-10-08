import tkinter as tk
from game import Game
from ui_tk import KillerSudokuApp
from generator import SudokuGenerator

def main():
    g = SudokuGenerator()
    cages, ans = g.puzzle()
    root = tk.Tk()
    game = Game(starters={}, cages=cages, ans=ans)
    KillerSudokuApp(root, game)
    root.mainloop()

if __name__ == "__main__":
    main()