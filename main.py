import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_prep import *
from model import *

model = SimpleGNN()
training_data = generateTrainingData(GRID_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

loss_history = []

for epoch in range(EPOCH):
    random.shuffle(training_data)
    total_loss = None
    n = len(training_data)
    for data in training_data:
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(H1(output), data.y)
        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss
        loss.backward()
        optimizer.step()

    average_loss = total_loss.item() / n
    loss_history.append(average_loss)

    if (epoch + 1) % 1 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {average_loss}')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

model_path = os.path.join(SAVE_DIR, "trained_model_2.pth")
torch.save(model.state_dict(), model_path)

plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()