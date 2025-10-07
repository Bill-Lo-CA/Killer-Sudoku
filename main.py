import tkinter as tk
from game import Game
from ui_tk import KillerSudokuApp
from generator import SudokuGenerator

def main():
    g = SudokuGenerator()
    cages = g.puzzle()
    
    root = tk.Tk()
    game = Game(starters={}, cages=cages)
    KillerSudokuApp(root, game)
    root.mainloop()

if __name__ == "__main__":
    main()