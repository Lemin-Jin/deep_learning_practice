import torch
from torch.utils.data import TensorDataset, DataLoader

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# return an iterator for 
def k_fold_data(dataset, k):
    X, y = dataset.data, dataset.targets
    fold_size = len(X) // k
    X_folds = [*X.split(fold_size)]
    y_folds = [*y.split(fold_size)]
    for i in range(k):
        train_X_dataset = torch.cat(X_folds[0 : i] + X_folds[i + 1 : ])
        train_y_dataset = torch.cat(y_folds[0 : i] + y_folds[i + 1 : ])
        validate_X_dataset = X_folds[i]
        validate_y_dataset = y_folds[i]
        train = TensorDataset(*(train_X_dataset.type(torch.float32), train_y_dataset))
        validate = TensorDataset(*(validate_X_dataset.type(torch.float32), validate_y_dataset))
        yield train, validate

def train_batch(net, batch, loss, updater):
    net.train()
    for X,y in batch:
        y_hat = net(X)
        l = loss(y_hat, y)
        l.mean().backward()
        updater.step()

def evaluate_accuracy(net, iter): 
    metric = Accumulator(2) 
    for X, y in iter:
        metric.add(accuracy(y, net(X)), y.numel())
    return metric[0] / metric[1]

def accuracy(y, y_hat):
        if len(y_hat.shape) > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

def train(net, dataset_iter, num_epochs, loss, updater, batch_size, loss_stat):
    
    for train, validate in dataset_iter:
        train_l_sum, valid_l_sum = 0, 0
        train_iter = DataLoader(train, batch_size, shuffle = True)
        validate_iter = DataLoader(validate, batch_size, shuffle = False)
        train_batch(net, train_iter, loss, updater)
        for epoch in range(num_epochs):
            train_batch(net, train_iter, loss, updater)
            train_loss = loss_stat(net, train_iter)
            test_loss = loss_stat(net, validate_iter)
            train_l_sum += (1 - train_loss)
            valid_l_sum += (1 - test_loss)
        print("train loss:" + str(train_l_sum/num_epochs))
        print("validation loss:" + str(valid_l_sum/num_epochs))

def k_fold_validate(data, k, net, loss, updater, num_epoch, batch_size, loss_stat):
    dataset_iter = k_fold_data(data, k)
    train(net, dataset_iter, num_epoch, loss, updater, batch_size, loss_stat)
