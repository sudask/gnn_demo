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

        print(f'\rEpoch: {epoch+1:3d} | LR: {current_lr:.6f} | Loss: {loss.item():.6f}', end='')

    print()
    end = time.time()
    print(f"Time: {end-start:.3f}")
    torch.save(model.state_dict(), "checkpoints/test.pth")

    plt.plot(np.array(range(51, 201)), loss_history[50:])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()