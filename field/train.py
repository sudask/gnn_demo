from config import*

def trainAndPlot(model, training_data, optimizer, criterion):
    loss_history = []

    for epoch in range(200):
        random.shuffle(training_data)
        for data in training_data:
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, data.vals)

            loss.backward()
            optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 1 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), "checkpoints/test.pth")

    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()