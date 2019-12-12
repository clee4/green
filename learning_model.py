from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from torchviz import make_dot
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np # we always love numpy
import time
import random
import math

import tictactoe

class Mytictactoe(nn.Module):
    # The init funciton in Pytorch classes is used to keep track of the parameters of the model
    # specifically the ones we want to update with gradient descent + backprop
    # So we need to make sure we keep track of all of them here
    def __init__(self, learning_rate=.01):
        super(Mytictactoe, self).__init__()
        # learning rate
        self.lr = learning_rate

        # layers defined here
        # we'll use this activation function internally in the network
        self.activation_func = torch.nn.ReLU()

        input_size = 27
        fc1_size = 130
        fc2_size = 250
        fc3_size = 140
        fc4_size = 60
        fc5_size = 9

        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.fc5 = nn.Linear(fc4_size, fc5_size)

    # The forward function in the class defines the operations performed on a given input to the model
    # and returns the output of the model
    def forward(self, x):
        # Go through all the layers
        x = self.fc1(x)
        x = self.activation_func(x)
        x = self.fc2(x)
        x = self.activation_func(x)
        x = self.fc3(x)
        x = self.activation_func(x)
        x = self.fc4(x)
        x = self.activation_func(x)
        x = self.fc5(x)

        return x

    def get_loss(self):
        # Loss function
        loss = nn.MSELoss()
        # Optimizer, self.parameters() returns all the Pytorch operations that are attributes of the class
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return loss, optimizer
        
    
class play_and_train:
    def __init__(self, model, discount=.85, explore=.4):
        self.game = tictactoe.tictactoe()
        self.model = model
        
        # attempts to load a model and keep previous training
        try:
            self.load_model()
            print("model loaded")
        except:
            print("model not trained")

        # discount factor
        self.g = discount
        # exploration rate
        self.e = explore

        self.device = 'cpu'

    def save_model(self):
        # saves the weights from the model
        torch.save(self.model.state_dict(), "rl.pth")
    
    def load_model(self):
        # loads the weights from the file a model was saved at
        self.model.load_state_dict(torch.load('rl.pth'))

    def train_model(self, batch_size, num_batches):
        loss, optimizer = self.model.get_loss()
        # Tells you % completion
        for i in range(num_batches):
            if (100*(i+1)/num_batches % 5 == 0):
                print(100*(i+1)/num_batches, "% training completion")

            states = []
            rewards = []
            # List of states and games
            for _ in range(batch_size):
                game, winner = self.play_game()
                
                outcome, q_tables = self.get_reward(winner, game)

                states = states + outcome
                rewards = rewards + q_tables
            # Train Model
            # convert data type to tensor
            states = torch.FloatTensor(states)
            rewards = torch.FloatTensor(rewards)
            
            for j in range(len(states)):
                # send to cuda
                # Rewards are the labels, states are the inputs
                # print(states[j])
                inputs = Variable(states[j]).to(self.device)
                labels = Variable(rewards[j]).to(self.device)
                
                optimizer.zero_grad()

                #Forward ,'\n', labels)
                outputs = self.model(inputs)

                # Compute the loss and find the loss with respect to each parameter of the model
                loss_size = loss(outputs, labels)
                loss_size.backward()

                # Change each parameter with respect to the recently computed loss.
                optimizer.step()

    def get_reward(self, winner, game):
        """
        Rewards the player based on their move and the current board position when the move was made

        game - [[[current board], index of next move], all the moves in the game]
        winner = x,o,empty
        """
        # reward_multiplier establishes the rewards for each player given win case
        reward_multiplier = {"x":2, "o":2}

        if winner is "x":
            reward_multiplier = {"x":1, "o":-1}
        elif winner is "o":
            reward_multiplier = {"x":-1, "o":1}

        # lists to hold the entire games positions and corresponding q_table
        states = []
        q_tables = []

        for i in range(len(game)):
            # x goes first; for all the x moves multiplier is different
            player = "x" if i%2 == 0 else "o"
            reward_vector = np.zeros(9)
            reward_vector[game[i][-1]] = reward_multiplier[player]*(self.g**(math.floor((len(game) - i) / 2) - 1))

            # Append the one hot version of the game state and the corresponding q_table
            states.append(self.game.one_hot(game[i][0]))
            q_tables.append(reward_vector)    
            
        return (states, q_tables)
    
    def play_game(self, training=True):
        """
        return list of game states and which move was made at each turn
        """
        # clears the grid
        self.game.clear_grid()
        player = ['x', 'o']

        states = []
        
        # Plays a game
        for i in range(9):
            # decides if the player will explore new moves this turn
            explore = True if random.random() < self.e else False
            # defines which player is playing based on even or odd turn
            turn = player[i%2]
            state = self.game.get_grid()
            
            # while training, we want new moves
            if explore and training:
                move = self.random_move()

            # while not training or exploring, we just want the best move
            else:
                # gets best move
                move = self.greedy_move(state)
            
            # appends the states to add to q table
            states.append([state, move])

            #updates grid and checks win case
            winner = self.game.update_grid(move, turn)

            # breaks if there is a winner before all tiles are filled
            if winner is not "":
                break
        
        return (states, winner)

    def play_human(self, order):
        """
        Plays a game against a human with input
        """
        self.game.clear_grid()
        player = ['x', 'o']


        # if the human is going first:
        if order%2 == 1:
            for i in range(9):
                state = self.game.get_grid()
                print(self.game)
                turn = player[i%2]

                if i%2 == 0:
                    move = int(input("Enter your move 1 through 9 going from left to right, top to bottom\n"))-1
                else:    
                    move = self.greedy_move(state)

                winner = self.game.update_grid(move, turn)

                if winner is not "":
                    break
            return winner
        
        # If the human is going second
        else:
            for i in range(9):
                state = self.game.get_grid()
                print(self.game)
                turn = player[i%2]

                if i%2 == 1:
                    move = int(input("Enter your move 1 through 9 going from left to right, top to bottom\n"))-1
                else:    
                    move = self.greedy_move(state)

                winner = self.game.update_grid(move, turn)

                if winner is not "":
                    break
            
            return winner

    def greedy_move(self, state):
        """
        Selects best move based on q table

        returns index of selected move
        """
        # Uses the model with the one_hot state to find all move probabilities
        tensor_grid = torch.FloatTensor(self.game.one_hot(state))
        output = self.model(Variable(tensor_grid).to(self.device))
        
        output = output.detach().numpy()

        # Make sure that the move is legal
        while True:
            # finds highest probability (best move) unless illegal
            best_move = np.argmax(output)
            if state[best_move] is "":
                return best_move
            else:
                # make the best move undesirable if it is illegal
                output[best_move] = -1000

    def random_move(self):
        """
        Selects a random legal move
        
        returns index of selected move
        """
        grid = self.game.get_grid()

        move_list = []
        # Append all legal moves to a move list
        for i in range(9):
            if grid[i] is "":
                move_list.append(i)
        
        # Choose a random move from legal moves
        return random.choice(move_list)

if __name__ == "__main__":
    net = Mytictactoe()
    
    player = play_and_train(net, explore=.3)
    player.train_model(500, 500)

    player.play_game(True)

    player.save_model()