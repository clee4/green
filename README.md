# Using Evolutionary AI and Reinforcement Learning to learn Tic-Tac-Toe
### Chris Lee and David Tarazi

## Our Goals
We wanted to:

* Learn and implement evolutionary AI and reinforcement learning
* Compare the different runtimes and effectiveness of these learning methods
* Learn how to save and pit models against each other
* Learn how to pluck out and mutate different weights within a net
* Figure out how we can edit the explore rate (RL), generation size (EV), and mutation rate (EV) to enhance learning

## Reinforcement Learning
When we went to implement reinforcement learning, we decided to use a strategy called deep q learning. This strategy is a way of taking all the possible states for a system (in our case a game of Tic-Tac-Toe) and map a reward to each possible move from that state. Therefore, we created a mathematical function that takes in a state and a potential action and returns a corresponding reward that has been built through previous experience (training). However, the way we implemented this due to the massive number of possible game states, was by doing q-learning on a network where we created a deep network with an input of a game state and an output of the rewards corresponding to each action. We then trained the weights over many games where the network played against itself to show it the potential ways of receiving rewards and to tweak the weights in order to acheive the best rewards. 

In the end, we were able to create a pretty solid network with an explore rate of 30% that was very good at playing against itself. However, the training took a very long time (almost 10 hours) and in the end, it was only really effective against itself, not human players. We could have tweaked the way the model trained in a few ways to make it better. For instance, we could have created or downloaded a good Tic-Tac-Toe player that would introduce only the best strategies to the bot to help it learn quicker and a more diverse range of strategies without relying on a random play from the explore rate. Furthermore, we could have created a way for the explore rate to be very high in the beginning of training (when the model doesn't know any rewards) that drops off as the model knows most of the strategies and is just trying to optimize the best strategy. Had we implemented these functions, we believe the gameplay for the model would have been better. 

## Evolutionary AI
On the other hand, we wanted to learn how a bio-inspired learning network worked which took in a model and made random iterations to a random number of weights within the simple, 2 layer network. The way this implementation works is that every generation of k players will play a game as the 1st and 2nd turn against every other player and get a reward for winning or tying. After comparing every players score, the player with the maximum score against all other players becomes the parent for the next generation. Then, that parent has a bunch of random mutations done to its weights to create a new generation of k players which will repeat the cycle for some number of generations to train. We hoped that this method would eventually converge to a parent that could easily beat or tie every other child. In the end after 2000 generations, this hope was true, but not in the exact way that we wanted it. While the player could beat every other player except for a few, these other players had predictable strategies that human players didn't always follow. As a result, the actual effectiveness of the model was very poor similar to the reinforcement learning model.

Had we enabled the bot to play against a few good players to train instead of playing against bad players, then the model's performance would likely have increased. We also suspect that our mutations were not as accurate as we might have wanted. When a weight got mutated, the new weight was assigned a random value between the min and max of all the weights instead of slightly changing it. Due to the large number of weights it is very unlikely to randomly get the right weights to converge. We also were only taking the top player instead of combining the top few players and finding some combination of their weights.

## Testing
When the bots play against each other, the same outcome happens every time, and the reinforcement learning model always wins. Not only was this model quicker, it was also a better performing model and learned an optimal winning strategy. However, it appears as though this strategy was not developed well enough because it doesn't adapt very well to a human making a move it isn't used to. Thus, we could have used some more random manipulation and introduced more strategies to optimize the networks. 

