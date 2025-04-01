from config import *

def trainAndPlot(model, training_data, optimizer, criterion):
    loss_history = []

    for epoch in range(EPOCH):
        random.shuffle(training_data)
        total_loss = None
        n = len(training_data)
        for data in training_data:
            optimizer.zero_grad()
            output = model(data)

            if USE_OBS:
                loss = criterion(observeOperator(output, data.label[1]), data.label[0])
            else:
                loss = criterion(output, data.label[0])

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            if loss > 0.01:
                loss.backward()
                optimizer.step()

        average_loss = total_loss.item() / n
        loss_history.append(average_loss)

        if (epoch + 1) % 1 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {average_loss}')

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    model_path = os.path.join(SAVE_DIR, PTH_FILE_NAME)
    torch.save(model.state_dict(), model_path)

    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

