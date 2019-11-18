import learning_model

class tictactoe:
    def __init__(self, iterations, learning_rate = 0.1, discount = 0.95):
        
        self.grid = [""] * 9

        self.learning_rate = learning_rate
        self.discount = discount # How much we appreciate future reward over current
        self.exploration_rate = 1.0 # Initial exploration rate
        self.exploration_delta = 1.0 / iterations # Shift from exploration to explotation

    # Tic Tac Toe Mechanics
    def __str__(self):
        """
        Formats the print to print a grid i.e.
        -  x  -
        -  x  -
        -  x  -

        returns string
        """
        grid = "GRID STATE\n"
        for i in range(0, 9, 3):
            for j in range(i, i+3):
                if self.grid[j] is "":
                    grid = grid + "-  "
                else:
                    grid = grid + self.grid[j] + "  "
            grid = grid + "\n"
        
        return grid

    def get_grid(self):
        """
        returns list with state of the grid
        """
        return self.grid

    def clear_grid(self):
        """
        Clears the grid to all empty strings (like starting state)
        """
        self.grid = [""] * 9
    
    def update_grid(self, index, move):
        """
        Checks to make sure the move is legal, then executes the move and checks if a win case was reached
        
        index: position to place move
        move: x or an o
        
        returns "x" or "o" to indicate which won if a win case was reached, otherwise nothing is returned
        """
        self.grid[index] = move.lower()
        
        win = self.check_win()
        if win is not "":
            print(self.__str__())
        return win

    def one_hot(self):
        """
        One hot encodes current grid state

        """
        mapp = {"":[1,0,0], "o":[0,1,0], "x":[0,0,1]}
        one_hot_grid = []
        
        for i in range(9):
            one_hot_grid.append(mapp[self.grid[i]])

        return one_hot_grid

    def check_win(self):
        """
        Checks for win in grid
        
        returns "x" or "o" if win
        returns "" if no win
        """
        checks = [self.check_columns(), 
                  self.check_diags(), 
                  self.check_rows()]
        
        for check in checks:
            if check is not "":
                return check
        
        return ""

    def check_diags(self):
        """
        Checks for win in diagonals
        
        returns "x" or "o" if win
        returns "" if no win
        """
        if self.grid[0] == self.grid[4] and self.grid[0] == self.grid[8] and len(self.grid[0]) > 0:
            return self.grid[0]
        elif self.grid[2] == self.grid[4] and self.grid[2] == self.grid[6] and len(self.grid[2]) > 0:
            return self.grid[2]  
        return ""

    def check_rows(self):
        """
        Checks for win in rows
        
        returns "x" or "o" if win
        returns "" if no win
        """
        # Row
        for i in range(0,9,3):
            # if the entire row is the same
            if len(self.grid[i]) > 0:  
                if self.grid[i] == self.grid[i+1] and self.grid[i] == self.grid[i+2]:
                    return self.grid[i]
        return ""
                    

    def check_columns(self):
        """
        Checks for win in columns
        
        returns "x" or "o" if win
        returns "" if no win
        """
        for i in range(3):
            if len(self.grid[i]) > 0:  
                if self.grid[i] == self.grid[i+3] and self.grid[i] == self.grid[i+6]:
                    return self.grid[i]
        return ""




    # Learning Aspect
    def get_reward(self):

    def one_hot(self):
        """
        One hot encodes current grid state

        """
        mapp = {"":[1,0,0], "o":[0,1,0], "x":[0,0,1]}
        one_hot_grid = []
        
        for i in range(9):
            one_hot_grid.append(mapp[self.grid[i]])

        return one_hot_grid

    def select_random_move(board):
        while True:
            c = random.randint(0, 8)
            if board[c] == 0:
                return c

    def select_strategic_move(board, model):


    def play_and_train(model, model_2, games_until_training=1000, total_games=8000, e_greedy=.7):
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        games = []
        current_game = []

        for i in range(0, total_games):
            playing = True
            nn_turn = True
            board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            # sides --> -1 = Os, 1 = Xs
            current_game = []
            current_game.append(board.copy())

            while playing:
                if nn_turn:
                    if random.uniform(0, 1) <= e_greedy:
                        board[select_random_move(model,board)]=1
                        current_game.append(board.copy())
                    else:
                        board[select_smart_move(model, board)] = 1
                        current_game.append(board.copy())

                else:
                    if random.uniform(0, 1) <= e_greedy:
                        board[select_random_move(model_2,board)] = -1
                        current_game.append(board.copy())
                    else:
                        board[select_smart_move(model_2, board)] = -1
                        current_game.append(board.copy())

                playable = False

                for square in board:
                    if square == 0:
                        playable = True

                if not get_outcome(board) == 0:
                    playable = False

                if not playable:
                    playing = False

                nn_turn = not nn_turn

            games.append(current_game)


if __name__ == "__main__":
    # asdf = tictactoe()
    # asdf.update_grid([1,'x'])
    # asdf.update_grid([4,'x'])
    # asdf.update_grid([7,'x'])
    # asdf.clear_grid()
    
    # asdf.update_grid([2,'o'])
    # asdf.update_grid([4,'o'])
    # asdf.update_grid([6,'o'])

    # asdf.clear_grid()
    
    # asdf.update_grid([0,'o'])
    # asdf.update_grid([1,'o'])
    # asdf.update_grid([2,'o'])

