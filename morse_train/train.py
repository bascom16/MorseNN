import data
import model

epochs = 30  # Number of times to iterate over the entire dataset
losses = []

for epoch in range(epochs):
    for batch_x, batch_y in data.train_loader:
        # Forward pass
        outputs = model.model(batch_x)
        loss = model.criterion(outputs, batch_y)

        # Backward and optimize
        model.optimizer.zero_grad() # Clear previous gradients
        loss.backward()             # Compute gradients
        model.optimizer.step()      # Update weights

    losses.append(loss.item())  # Store the loss value

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')