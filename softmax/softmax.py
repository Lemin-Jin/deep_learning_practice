import torch 
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch import nn
from helper import *

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

class SoftMax_Trainer:
    def __init__(self, train_data, test_data, num_epochs, batch_size, learn_rate):
        self.train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle = True)
        self.test_iter = torch.utils.data.DataLoader(test_data, batch_size, shuffle = True)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        num_inputs = train_data[0][0].numel()
        num_outputs = len(torch.unique(train_data.targets))
        self.W = torch.normal(0, 0.01, size=(num_inputs,num_outputs), requires_grad= True)
        self.b = torch.zeros(num_outputs, requires_grad=True, dtype=float)


    def evaluate_accuracy(self, data_iter): 
        metric = Accumulator(2) 
        for X, y in data_iter:
            metric.add(self.accuracy(y, self.net(X)), y.numel())
        return metric[0] / metric[1]
    
    def cross_entropy(self, y, y_hat):
        return - torch.log(y_hat[range(len(y_hat)), y])

    def net(self, X):
        return self.Softmax(torch.matmul(X.reshape((-1, self.W.shape[0])), self.W) + self.b)
    
    def Softmax(self,X):
        # x.dim = (784, output) W.dim = 784 * len(output)
        # X*W.dim = batch_size * len(output)
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition  # 这里应用了广播机制

    def reset_grad(self,param):
        if param.grad != None:
            param.grad.zero_()

    def Stochastic_Gradient_Descent(self, params):
        with torch.no_grad():
            for param in params:
                param -= param.grad * self.learn_rate / self.batch_size
                self.reset_grad(param)

    def accuracy(self, y, y_hat):
        if len(y_hat.shape) > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    def training(self):
        metric = Accumulator(3)
        for X,y in self.train_iter:
            y_hat = self.net(X) 
            loss = self.cross_entropy(y, y_hat)
            loss.sum().backward()
            self.Stochastic_Gradient_Descent([self.W,self.b])

            metric.add(float(loss.sum()), self.accuracy(y, y_hat), y.numel())
        return metric[0] / metric[2], metric[1] / metric[2]
        
    def train(self): 
        for epoch in range(self.num_epochs):
            train_metrics = self.training()
            test_acc = self.evaluate_accuracy(self.test_iter)
            train_loss, train_acc = train_metrics
            print("@epoch " + str(epoch))
            print(train_loss, train_acc, test_acc)
    
        


train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

batch_size = 256
learn_rate = 0.1
num_inputs = train_data[0][0].numel()
num_outputs = len(torch.unique(train_data.targets))

net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
updater = torch.optim.SGD(net.parameters(), lr=0.05)

k_fold_validate(train_data, 5, net, loss, updater, 20, batch_size, evaluate_accuracy)

# smt = SoftMax_Trainer(train_data, test_data, 10, batch_size, learn_rate)
# smt.train()

