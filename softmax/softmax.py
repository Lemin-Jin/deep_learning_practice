import torch 
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

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

num_inputs = train_data[0][0].numel()
num_outputs = len(torch.unique(train_data.targets))
batch_size = 512
num_epoch = 10
learn_rate = 0.05

W = torch.normal(0, 0.01, size=(num_inputs,num_outputs), requires_grad= True)
b = torch.zeros(num_outputs, requires_grad=True)
test_iter = torch.utils.data.DataLoader(test_data, batch_size, shuffle = False)

def evaluate_accuracy(data_iter): 
    metric = Accumulator(2) 
    for X, y in data_iter:
        metric.add(accuracy(y, Softmax(X, W, b)), y.numel())
    return metric[0] / metric[1]

def Softmax(X, W, b):
    # x.dim = (784, output) W.dim = 784 * len(output)
    # X*W.dim = batch_size * len(output)
    O = torch.matmul(X.reshape((-1, W.shape[0])),W) + b
    y = torch.exp(O)
    return y/(y.sum(1, keepdims=True))

def cross_entropy(y, y_hat):
    return -torch.log(y_hat[range(len(y_hat)), y])



def reset_grad(param):
    if param.grad != None:
        param.grad.zero_()

def Stochastic_Gradient_Descent(params,learn_rate, batch_size):
    with torch.no_grad():
        for param in params:
            param -= param.grad * learn_rate / batch_size
            reset_grad(param)

# for test value


def accuracy(y, y_hat):
    if len(y_hat.shape) > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def training(train_data):
    num_outputs = len(torch.unique(train_data.targets))
    batches = torch.utils.data.DataLoader(train_data, batch_size, shuffle = True)
    acc = [0,0,0]
    for X,y in batches:
        y_hat = Softmax(X, W, b)
        loss = cross_entropy(y, y_hat)
        loss.sum().backward()
        Stochastic_Gradient_Descent([W,b], learn_rate, batch_size)
        acc = [i + j for i, j in zip(acc, [loss.sum(), accuracy(y, y_hat), y.numel()])]
        return acc[0]/acc[2], acc[1]/acc[2]

def train_ch3(train_data, test_iter, num_epochs): 
    for epoch in range(num_epochs):
        train_metrics = training(train_data)
        test_acc = evaluate_accuracy(test_iter)
        print("@epoch " + str(epoch))
        print(train_metrics, test_acc)

train_ch3(train_data, test_iter, num_epoch)


