import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    # Define a neural network for Q-value approximation
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the neural network
        super(Linear_QNet, self).__init__()
        # Define the layers of the neural network
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # Define the forward pass through the network
    def forward(self, x):
        # Pass the input through the first linear layer and apply ReLU activation function
        x = F.relu(self.linear1(x))
        # Pass the output of the first layer through the second linear layer
        x = self.linear2(x)
        return x

    # Save the model parameters to a file
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        # Create the model folder if it doesn't exist
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        # Save the model parameters to the specified file
        full_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), full_path)


class QTrainer:
    # Class to handle the training process
    def __init__(self, model, lr, gamma):
        # Initialize training parameters
        self.learning_rate = lr
        self.gamma = gamma
        # Set the model for training
        self.model = model
        # Initialize the optimizer using Adam algorithm for gradient descent optimization
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        # Define the loss function
        self.loss_function = nn.MSELoss()

    # Perform a single step of training
    def train_step(self, state, action, reward, next_state, done):
        # Convert input data to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Unsqueeze if single instance (1D tensor)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Predict Q-values with the current state
        pred_q_values = self.model(state)
        # Create a copy of predicted Q-values
        target_q_values = pred_q_values.clone()
        
        # Update Q-values using the Bellman equation
        for idx in range(len(done)):
            updated_q_value = reward[idx]
            if not done[idx]:
                updated_q_value += self.gamma * torch.max(self.model(next_state[idx]))
            target_q_values[idx][torch.argmax(action[idx]).item()] = updated_q_value
        
        # Train the model
        self.optimizer.zero_grad()
        # Compute the loss between predicted and target Q-values
        loss = self.loss_function(target_q_values, pred_q_values)
        # Backpropagate the loss and update model parameters
        loss.backward()
        self.optimizer.step()
