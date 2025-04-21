from config import*

def trainAndPlot(model, training_data, optimizer, scheduler, criterion):
    start = time.time()
    loss_history = []

    for epoch in range(200):
        random.shuffle(training_data)
        loss = 0
        for data in training_data:
            optimizer.zero_grad()
            output = model(data)

            loss += criterion(output, data.vals)

        loss /= len(training_data)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 1 == 0:
            print(f'Epoch: {epoch + 1}')
            print(f'Learning rate = {current_lr}')
            print(f'Loss: {loss.item()}')

    end = time.time()
    print(f"Training time: {end-start}")
    torch.save(model.state_dict(), "checkpoints/test.pth")

    plt.plot(np.array(range(51, 201)), loss_history[50:])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()