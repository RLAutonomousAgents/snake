# SnakeGameAI
The project was initiated to adapt a reinforcement learning approach to tackle the Snake game, enabling the software to autonomously learn how to play through trial and error. It consists of three primary components:

1. `agent.py`: This script implements the Q-learning-based agent responsible for playing the Snake game. The `SnakeAgent` class orchestrates the agent's interactions with the game environment, employing a straightforward linear neural network for decision-making. The agent learns from experiences, updating both short-term and long-term memory to determine actions. The training loop manages the agent's gameplay, training sessions, and progress monitoring.

2. `game.py`: This module defines the Snake game and its Pygame-based mechanics. It encompasses methods for initializing, resetting, executing steps, detecting collisions, and updating the user interface. Additionally, it controls the snake's movement and direction based on the agent's chosen action.

3. `linear_qnet.py`: Utilizing PyTorch, this code establishes the Q-learning neural network model and its associated trainer. The neural network predicts Q-values, which are utilized in the Bellman equation to compute target Q-values. A loss function gauges the disparity between predicted and target Q-values, while backpropagation adjusts the neural network parameters to minimize this discrepancy, thereby refining the Q-values estimate.

The project employs the deep Q-network (DQN), a model-free reinforcement learning algorithm that integrates deep learning techniques with Q-learning to approximate the Q-function. Within the code, adjustments to hyperparameters such as the learning rate and discount factor were made to seek the optimal solution, yielding highly promising results.
