from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from torchviz import make_dot
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np # we always love numpy
import time
import random
import math

import tictactoe

class Player(nn.Module):
    # The init funciton in Pytorch classes is used to keep track of the parameters of the model
    # specifically the ones we want to update with gradient descent + backprop
    # So we need to make sure we keep track of all of them here
    def __init__(self):
        super(Player, self).__init__()
        # layers defined here
        # we'll use this activation function internally in the network
        self.activation_func = torch.nn.ReLU()

        input_size = 27
        fc1_size = 27
        fc2_size = 9

        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)

    # The forward function in the class defines the operations performed on a given input to the model
    # and returns the output of the model
    def forward(self, x):
        # Go through all the layers
        x = self.fc1(x)
        x = self.activation_func(x)
        x = self.fc2(x)

        return x

def greedy_move(player, one_hot_state, state, device="cpu"):
    """
    Selects best move based on q table

    returns index of selected move
    """
    tensor_grid = torch.FloatTensor(one_hot_state)
    output = player(Variable(tensor_grid).to(device))
    
    output = output.detach().numpy()

    # Make sure that the move is legal
    while True:
        #print(output, "\n", state)
        best_move = np.argmax(output)
        # print(output)
        # print(best_move)
        if state[best_move] is "":
            return best_move
        else:
            # make the best move undesirable if it is illegal
            output[best_move] = -1000

def make_children(player, birth_defect_rate=.25, num_children=100):
    """
    makes a bunch of offspring (boomers) based on a player

    player: player object to base children off of
    birth_defect_rate: rate at which weights are randomly modified
    num_children: number of boomers to make

    returns list of children
    """
    # list to hold offspring of player
    boomers = []

    # finds max and minimum weights
    with torch.no_grad():
        max_w = float(torch.max(player.fc1.weight))
        min_w = float(torch.min(player.fc1.weight))

    # iterates through to make n children
    for _ in range(num_children):
        
        # creates a copy of the player
        child = deepcopy(player)
        
        with torch.no_grad():
            # iterates through layer 1 weights
            for i in range(len(child.fc1.weight)):
                for j in range(len(child.fc1.weight[0])):
                # whether or not to give child random mutation
                    if random.random() <= birth_defect_rate:
                        child.fc1.weight[i,j] = random.uniform(min_w, max_w)
            
            # iterates through layer 2 weights
            for i in range(len(child.fc2.weight)):
                for j in range(len(child.fc2.weight[0])):
                # whether or not to give child random mutation
                    if random.random() <= birth_defect_rate:
                        child.fc2.weight[i,j] = random.uniform(min_w, max_w)
        
        boomers.append(child)
        
    return boomers


def tournament(players, game):
    scores = [0]*len(players)
    # first iteration players all go first agaisnt opponents
    first = True
    # select player to play against rest
    for _ in range(2):
        for i in range(len(players)):
            # iterate through the rest of the players and count total score (wins, ties) against all the others
            for opponent in players:
                if players[i] == opponent: 
                    pass 
                else:
                    scores[i] += play_game(players[i], opponent, first, game)
        # change order so all players play again, but going second this time
        first = False    

    # parent is the player with the maximum score against all other players
    print(max(scores))
    parent = players[scores.index(max(scores))]
    return parent


def play_game(player1, player2, first, game):
    
    game.clear_grid()

    who_goes = 1
    turn = ['o', 'x']
    if first:
        turn = ['x', 'o']
        who_goes = 0
    
    for i in range(9):
        # gets the one hot current grid
        one_hot = game.one_hot()
        state = game.get_grid()
        
        if i%2 == who_goes:
            move = greedy_move(player1, one_hot, state)
        else:
            move = greedy_move(player2, one_hot, state)

        winner = game.update_grid(move, turn[i%2])

        if winner is not "":
            break

    # maps a winner to a score
    scores = {'x':1, '':1, 'o':-1}
    return scores[winner]

def save_model(player):
    torch.save(player.state_dict(), "ev.pth")

def load_model(player):
    player.load_state_dict(torch.load('ev.pth'))
    return player

if __name__ == "__main__":
    player = Player()
    try:
        player = load_model(player)
        print("Model loaded")
    except:
        print("Model not trained")

    game = tictactoe.tictactoe()
    for i in range(50):
        print(i)
        print(game)
        players = make_children(player, num_children=40)
        player = tournament(players, game)
    
    save_model(player)
    

    