from data_loader import *
from model import *

model = SimpleGNN()
training_data = prepare_training_data()
testint_data = prepare_testing_data()

optimizer = optim.Adam(model.parameters(), lr = LR)
criterion = nn.MSELoss()

for epoch in range(EPOCH):
    total_loss = None
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
            print(f'Epoch: {epoch + 1}, Loss: {total_loss.item()}')
            
    if epoch + 1 == EPOCH:
        for data in training_data:
            print("real:", data.y, "approx:", model(data))