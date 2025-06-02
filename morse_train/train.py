import data
import model

epochs = 30  # Number of times to iterate over the entire dataset
losses = []

for epoch in range(epochs):
    # Forward pass
    outputs = model.model(data.xtrain_tensor)
    loss = model.criterion(outputs, data.ytrain_tensor)

    # Backward and optimize
    model.optimizer.zero_grad() # Clear previous gradients
    loss.backward()             # Compute gradients
    model.optimizer.step()      # Update weights

    losses.append(loss.item())  # Store the loss value

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')