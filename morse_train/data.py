import numpy as np

from numpy import load

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



