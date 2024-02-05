import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from linear_qnet import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
from IPython import display

# Define the game settings and hyperparameters for training
MAX_MEMORY = 100000 # Maximum number of experiences we are storing
BATCH_SIZE = 1000 # Number of experiences we use for training per batch
LEARNING_RATE = 0.001 # Learning rate used by the optimizer

plt.ion() 

# Plot the training progress (scores and mean scores)
def plot_training_progress(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)


class SnakeAgent:
    # Initialize the agent class with the specified settings
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(15, 256, 3) # 15 inputs (game states), 256 hidden layers, 3 outputs (actions)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

# Get current state of the game as a vector
    def get_game_state(self, game):
        head = game.snake[0]
        # where the danger is
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # directions
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # check for the presence of the snake's body segments in neighboring cells
        body_left = any(segment == point_l for segment in game.snake[1:])
        body_right = any(segment == point_r for segment in game.snake[1:])
        body_up = any(segment == point_u for segment in game.snake[1:])
        body_down = any(segment == point_d for segment in game.snake[1:])

        # define the state as a list of binary values 
        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            # snake's own body
            body_left,
            body_right,
            body_up,
            body_down
        ]

        return np.array(state, dtype=int)

    # Add the current game state to the memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Train the agent using the long-term memory
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) # unzip the tuples in list to multiple tuples
        self.trainer.train_step(states, actions, rewards, next_states, dones) # train our model

    # Train the agent using the short-term memory
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # Get the action to take based on the current game state using the epsilon-greedy method
    def get_action(self, state):
        self.epsilon = 80 - self.number_of_games # epsilon value decreases as the number of games increases
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon: # with a probability of epsilon/200, the agent will take a random action
            move = random.randint(0, 2) 
            final_move[move] = 1
        else:
            # Convert the current state of the game into a PyTorch tensor
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            # Get the model's predictions for the current state
            prediction = self.model(state0)
            # Get the action with the highest predicted value
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

# Train the agent
def train_agent():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    highest_score = 0
    agent = SnakeAgent()
    game = SnakeGameAI()

    while True:
        # get old state and action from the environment to train the agent
        state_old = agent.get_game_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_game_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        # train the agent using the long-term memory
        if done:
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > highest_score:
                highest_score = score
                agent.model.save()
    
            print(f'Game: {agent.number_of_games}, Score: {score}, Record: {highest_score}')
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot_training_progress(plot_scores, plot_mean_scores)
        

# Run the training
if __name__ == '__main__':
    train_agent()
