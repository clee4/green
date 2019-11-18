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
from tictactoe import tictactoe

class Mytictactoe(nn.Module):
    # The init funciton in Pytorch classes is used to keep track of the parameters of the model
    # specifically the ones we want to update with gradient descent + backprop
    # So we need to make sure we keep track of all of them here
    def __init__(self):
        super(Mytictactoe, self).__init__()
        # learning rate
        self.lr = .001

        # discount factor
        self.gamma = .85

        # exploration rate
        self.e = .3

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

    def get_reward(self):
        pass
    
class play_and_train:
    def __init__(self, model):
        self.game = tictactoe()
        self.model = model

    def play_game(self):
        """
        return list of game states
        """
        self.game.clear_grid()
        player = ['x', 'o']

        states = []
        for i in range(9):
            explore = True if random.random() < self.model.e else False
            turn = player[i%2]

            if explore:
                move = self.random_move()
            else:
                move = self.greedy_move()
            
            states.append(self.game.get_grid())

            # this will need to be tweaked to work better for rewards
            if self.game.update_grid(move, turn) is not "":
                break
        
        return states
    
    def greedy_move(self):
        """
        Selects best move based on q table

        returns index of selected move
        """
        pass

    def random_move(self):
        """
        Selects a random legal move
        
        returns index of selected move
        """
        grid = self.game.get_grid()

        move_list = []
        for i in range(9):
            if grid[i] is "":
                move_list.append(i)
        
        return random.choice(move_list)
