from config import*

def train(model, training_data, validation_data, optimizer, scheduler, criterion, num_epoch, save_path):
    start = time.time()
    loss_history = []

    for epoch in range(num_epoch):
        model.train()
        random.shuffle(training_data)
        epoch_loss = 0.0

        for data in training_data:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.vals)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(training_data)
        loss_history.append(epoch_loss)

        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for data in validation_data:
        #         outputs = model(data)
        #         val_loss += criterion(outputs, data.vals).item()
        
        # val_loss /= len(validation_data)
        
        # scheduler.step(val_loss)

        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\rEpoch: {epoch+1:3d} | LR: {current_lr:.6f} | '
              f'Train Loss: {epoch_loss:.6f}', end='')

    print()
    end = time.time()
    print(f"Time: {end-start:.3f}s")
    torch.save(model.state_dict(), save_path)

    return loss_history