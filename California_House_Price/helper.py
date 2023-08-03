import torch
from torch.utils.data import TensorDataset, DataLoader

# class Accumulator:  #@save
#     """在n个变量上累加"""
#     def __init__(self, n):
#         self.data = [0.0] * n

#     def add(self, *args):
#         self.data = [a + float(b) for a, b in zip(self.data, args)]

#     def reset(self):
#         self.data = [0.0] * len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# # return an iterator for 
# def k_fold_data(train_label, train_feature, k):
#     X, y = train_feature, train_label
#     fold_size = len(X) // k
#     X_folds = [*X.split(fold_size)]
#     y_folds = [*y.split(fold_size)]
#     for i in range(k):
#         train_X_dataset = torch.cat(X_folds[0 : i] + X_folds[i + 1 : ])
#         train_y_dataset = torch.cat(y_folds[0 : i] + y_folds[i + 1 : ])
#         validate_X_dataset = X_folds[i]
#         validate_y_dataset = y_folds[i]
#         # train = TensorDataset(*())
#         # validate = TensorDataset(*(validate_X_dataset.type(torch.float32), validate_y_dataset))
#         yield train_X_dataset.type(torch.float32), train_y_dataset, validate_X_dataset.type(torch.float32), validate_y_dataset

# def train_batch(net, batch, loss, updater):
#     net.train()
#     for X,y in batch:
#         updater.zero_grad()
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         l.backward()
#         updater.step()

# def evaluate_accuracy(net, iter): 
#     metric = Accumulator(2) 
#     for X, y in iter:
#         metric.add(accuracy(y, net(X)), y.numel())
#     return metric[0] / metric[1]

# def accuracy(y, y_hat):
#         if len(y_hat.shape) > 1:
#             y_hat = y_hat.argmax(axis=1)
#         cmp = y_hat.type(y.dtype) == y
#         return float(cmp.type(y.dtype).sum())

# def train(net, dataset_iter, num_epochs, loss, updater, batch_size, loss_stat):
#     num = 1
#     k = 1
#     for train_features, train_labels, test_features, test_labels in dataset_iter:
#         train_loss_sum, test_loss_sum = 0, 0
#         train = TensorDataset(*(train_features, train_labels))
#         train_iter = DataLoader(train, batch_size, shuffle = True)
#         train_batch(net, train_iter, loss, updater)
        
#         for epoch in range(num_epochs):
#             train_batch(net, train_iter, loss, updater)
#             train_loss = loss_stat(net, train_features, train_labels, loss)
#             test_loss = loss_stat(net, test_features, test_labels, loss)
#             train_loss_sum += train_loss
#             test_loss_sum += test_loss
#             print("train loss at epoch " + str(num) + ": " + str(train_loss))
#             print("test loss at epoch " + str(num) + ": " + str(test_loss))
#             num += 1
#         print("\n fold - " + str(k) + " test_loss: " + str(test_loss_sum/num_epochs))
#         k += 1

# def k_fold_validate(train_label, train_feature, k, net, loss, updater, num_epoch, batch_size, loss_stat):
#     dataset_iter = k_fold_data(train_label, train_feature, k)
#     train(net, dataset_iter, num_epoch, loss, updater, batch_size, loss_stat)







def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, loss, loss_calc):
    train_ls, test_ls = [], []

    # print(train_labels.shape)
    train_iter = DataLoader(TensorDataset(*(train_features, train_labels)), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
        train_ls.append(loss_calc(net, train_features, train_labels, loss))
        print(loss_calc(net, train_features, train_labels, loss))
        if test_labels is not None:
            test_ls.append(loss_calc(net, test_features, test_labels, loss))
            print(loss_calc(net, test_features, test_labels, loss))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size, net, loss, loss_calc):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size, loss, loss_calc)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k