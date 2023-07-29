import torch
import numpy as np
import random



def generate_data(weights, offset, size, mean, std, error):

    X = torch.normal(mean, std, (size, len(weights)))
    # compose y = X*w + b
    y = torch.matmul(X, weights) + offset
    # add noise epsilon to y
    # y = X * w + b + e
    error = torch.normal(0,0.01, y.shape)
    y += error
    return X, y.reshape((-1, 1))

def model(weights, X, offset):
    return torch.matmul(X,weights) + offset

def loss_func(y, y_hat):
    size = 0
    try:
        y.size(dim=0)
    except IndexError:
        print("y seems to have the wrong size")
        return None

    loss = ((1/2) * (y.reshape(y_hat.shape) - y_hat)**2)

    return loss

def reset_grad(param):
    if param.grad != None:
        param.grad.zero_()

def Stochastic_Gradient_Descent(params,learn_rate):
    with torch.no_grad():
        for param in params:
            param -= param.grad * learn_rate
            reset_grad(param)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


w = torch.tensor([4.2, -2])
b = 7
features, labels = generate_data(w, b, 1000, 0, 1, 0.02)


learn_rate = 0.03
num_epochs = 3
batch_size = 10
w_hat = torch.normal(0, 0.01, size=(2,1), requires_grad= True)
b_hat = torch.zeros(1, requires_grad=True)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = model(w_hat,X,b_hat)
        loss = loss_func(y, y_hat)
        
        loss.sum().backward()
        Stochastic_Gradient_Descent([w_hat,b_hat],learn_rate)
    with torch.no_grad():
        train_loss = loss_func(y, model(w_hat,X,b_hat))
        print("loss at epoch " + str(epoch) + " is " + str(train_loss.mean()))





    
 
