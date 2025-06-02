import numpy as np
import torch

from numpy import load
from torch.utils.data import TensorDataset, DataLoader

data = load('morse-dataset/baseline.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])

xtrain = data['xtr']
ytrain = data['ytr']
xval = data['xva']
yval = data['yva']
xtest = data['xte']
ytest = data['yte']

debug = True

if debug:
    print("xtrain\n", xtrain, xtrain.shape)
    print("ytrain\n", ytrain, ytrain.shape)

    print("xval\n", xval, xval.shape)
    print("yval\n", yval, yval.shape)

    print("xtest\n", xtest, xtest.shape)
    print("ytest\n", ytest, ytest.shape)

xtrain_tensor = torch.from_numpy(xtrain).float()
ytrain_tensor = torch.from_numpy(ytrain).float()
xval_tensor = torch.from_numpy(xval).float()
yval_tensor = torch.from_numpy(yval).float()
xtest_tensor = torch.from_numpy(xtest).float()
ytest_tensor = torch.from_numpy(ytest).float()

# Create dataset and dataloader
batch_size = 32  # Set your desired batch size
train_dataset = TensorDataset(xtrain_tensor, ytrain_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)