import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Load training JSON data
with open('soundStats.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)

# Preprocess the training data
X = df.drop('playlist', axis=1)
y = df['playlist']

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Build the neural network
class playlistClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(playlistClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

input_size = X_train.shape[1]
num_classes = len(np.unique(y_train))
model = playlistClassifier(input_size, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the neural network
epochs = 50
batch_size = 32

for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_test).sum().item()
    accuracy = correct / len(y_test)
    print(f'Test accuracy: {accuracy}')


with open('TestStats.json', 'r') as file:
    new_data = json.load(file)
new_df = pd.DataFrame(new_data)

# Preprocess the new data
new_X = new_df.drop('playlist', axis=1)
new_X = scaler.transform(new_X)
new_X = torch.tensor(new_X, dtype=torch.float32)

# Make predictions on the new data
with torch.no_grad():
    new_outputs = model(new_X)
    new_probs = nn.functional.softmax(new_outputs, dim=1)
    new_probs, new_predicted = torch.max(new_probs, 1)
    new_predicted_labels = encoder.inverse_transform(new_predicted.numpy())

# Get the actual playlist labels from the new data
new_actual_labels = new_df['playlist']

# Set a threshold for prediction probabilities
threshold = 0.5

# Classify data points with probabilities below the threshold as "other"
for i in range(len(new_predicted_labels)):
    if new_probs[i] < threshold:
        new_predicted_labels[i] = "other"

# Print predicted and actual playlist labels for each data point
Total = 0
count = 0
for i in range(len(new_predicted_labels)):
    print(f'Data point {i+1}: predicted={new_predicted_labels[i]}, actual={new_actual_labels[i]}')
    if str(new_predicted_labels[i]) != 'other':
        Total += 1
    if str(new_predicted_labels[i]) + "Test" == str(new_actual_labels[i]):
        count += 1
print("Accuracy: " + str(count/Total))
print(str(len(new_predicted_labels) - Total) + " data points below threshold and sorted into other, " + str(Total) + " data points used.")





