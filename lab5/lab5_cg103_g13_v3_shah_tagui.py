import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Hyperparameters ---
LEARNING_RATE = [0.01, 0.1, 1][0]
BATCH_SIZE = [1, 64, 256][1]
NUM_EPOCHS = 10 # Fixed
NUM_HIDDEN_LAYERS = [0, 1, 2][1]  # Number of hidden layers
HIDDEN_WIDTH = [32, 64, 128][0]  # Width of hidden layers
SEED = 42  # For reproducibility
ACTIVATION_FUNCTION = [nn.ReLU(), nn.Sigmoid(), nn.Tanh()][2]  # Activation function

# --- Dataset (FashionMNIST) ---
INPUT_SIZE = 28 * 28  # 784
OUTPUT_SIZE = 10      # 10 classes

# --- Device Configuration ---
USE_CUDA = True
device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')

print(f"Learning Rate: {LEARNING_RATE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Number of Hidden Layers: {NUM_HIDDEN_LAYERS}")
print(f"Hidden Layer Width: {HIDDEN_WIDTH}")
print(f"Activation Function: {ACTIVATION_FUNCTION.__class__.__name__}")

# --- Data Loading and Preprocessing ---
print("\n--- Data Loading and Preprocessing ---")

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the FashionMNIST dataset
full_train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Split training data into training and validation sets
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

print(f"Full training dataset size: {len(full_train_dataset)}")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("DataLoaders created.")

# --- MLP Model Definition ---
print("\n--- MLP Model Definition ---")

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, num_hidden_layers, hidden_width, output_size):
        super(MultilayerPerceptron, self).__init__()
        self.input_size = input_size
        layers = []
        torch.manual_seed(SEED)  # For reproducibility
        np.random.seed(SEED)

        # Input layer
        if num_hidden_layers == 0:
            layers.append(nn.Linear(input_size, output_size))
        else:
            layers.append(nn.Linear(input_size, hidden_width))
            layers.append(ACTIVATION_FUNCTION)

            # Hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_width, hidden_width))
                layers.append(ACTIVATION_FUNCTION)

            # Output layer
            layers.append(nn.Linear(hidden_width, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the image
        x = x.reshape(-1, self.input_size)
        return self.network(x)

# Instantiate the model
model = MultilayerPerceptron(
    input_size=INPUT_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    hidden_width=HIDDEN_WIDTH,
    output_size=OUTPUT_SIZE
).to(device)
print(model)

print("\n--- Training Process ---")
print(f"Using Activation Function: {ACTIVATION_FUNCTION.__class__.__name__}")

# Optimiser
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"Using Optimiser: Adam with LR={LEARNING_RATE}")

# Lists to store metrics for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
step_losses = []

for epoch in range(NUM_EPOCHS):
    # --- Training Step ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Backward and optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        step_losses.append(loss.item()) # Store loss for current step
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = 100 * correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # --- Validation Step ---
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / len(val_loader)
    epoch_val_acc = 100 * correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

print("Training finished.")

# --- Testing ---
print("\n--- Testing Process ---")
model.eval()
with torch.no_grad():
    correct_test = 0
    total_test = 0
    test_loss = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

    final_test_loss = test_loss / len(test_loader)
    final_test_acc = 100 * correct_test / total_test
    print(f"Test Loss: {final_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_acc:.2f}%")

# --- Results and Visualisation ---
print("\n--- Results and Visualisation ---")

epochs_range = range(1, NUM_EPOCHS + 1)

# Plotting Accuracy
fig_accuracy = go.Figure()
fig_accuracy.add_trace(go.Scatter(x=list(epochs_range), y=train_accuracies, mode='lines+markers', name='Training Accuracy'))
fig_accuracy.add_trace(go.Scatter(x=list(epochs_range), y=val_accuracies, mode='lines+markers', name='Validation Accuracy'))
fig_accuracy.update_layout(
    title='Training and Validation Accuracy vs. Epochs',
    xaxis_title='Epochs',
    yaxis_title='Accuracy (%)',
    legend_title='Legend'
)
fig_accuracy.show()

# Plotting Step-wise Training Loss
fig_step_loss = go.Figure()
fig_step_loss.add_trace(go.Scatter(x=list(range(len(step_losses))), y=step_losses, mode='lines', name='Training Loss per Step'))
fig_step_loss.update_layout(
    title='Training Loss vs. Learning Steps',
    xaxis_title='Learning Step',
    yaxis_title='Loss',
    legend_title='Legend'
)
fig_step_loss.show()
