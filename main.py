from data_prep import *
from model import *

model = SimpleGNN()
training_data = generateTrainingData(GRID_SIZE)

optimizer = optim.Adam(model.parameters(), lr = LR)
criterion = nn.MSELoss()

for epoch in range(EPOCH):
    # random.shuffle(training_data)
    total_loss = None
    n = len(training_data)
    for data in training_data:
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, data.y)
        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 100 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {total_loss.item() / n}')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

model_path = os.path.join(SAVE_DIR, "trained_model.pth")
torch.save(model.state_dict(), model_path)
