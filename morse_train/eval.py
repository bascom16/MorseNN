import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import model as m
import data as d
import train as t

# Switch model to evaluation mode (important for layers like dropout or batchnorm, though not strictly necessary for this simple model)
m.model.eval()

with torch.no_grad(): # Disable gradient calculations for evaluation
    y_pred_train = m.model(d.xtrain_tensor)
    y_pred_test = m.model(d.xtest_tensor)

    train_loss = m.criterion(y_pred_train, d.ytrain_tensor)
    test_loss = m.criterion(y_pred_test, d.ytest_tensor)
    print(f'\nTraining Loss: {train_loss.item():.4f}')
    print(f'Test Loss: {test_loss.item():.4f}')

dummy_input = torch.randn(64,64)  # Dummy input for ONNX export

input_names = ["input"]
output_names = ["output"]

torch.save(m.model, "morse_model.pth")

torch.onnx.export(m.model, 
                  (dummy_input,),
                  "morse_model.onnx",
                  export_params=True, 
                  input_names=input_names, 
                  output_names=output_names,
                  opset_version=12)
print("Model exported to morse_model.onnx")