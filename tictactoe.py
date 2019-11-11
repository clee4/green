class tictactoe:
    def __init__(self):
        self.grid = [0] * 9

    def update_grid(self, move):
        self.grid[move[0]] = move[1]

    def check_win(self):
        pass

    def check_diags(self):
        pass

    def check_rows(self):
        pass

    def check_columns(self):
        pass