import numpy as np
import pandas as pd
import torch
from torch import nn
import os
from torch.utils.data import TensorDataset
from helper import *

os.chdir(os.getcwd())
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")



redundant_cols = ['Address', 'Summary', 'City', 'State']
for c in redundant_cols:
    del test_data[c], train_data[c]

large_vel_cols = ['Lot', 'Total interior livable area', 'Tax assessed value', 'Annual tax amount', 'Listed Price', 'Last Sold Price']
for c in large_vel_cols:
    train_data[c] = np.log(train_data[c]+1)
    if c!='Sold Price':
        test_data[c] = np.log(test_data[c]+1)


all_features = pd.concat((train_data.iloc[:,2:],test_data.iloc[:,1:]))

all_features['Listed On'] = pd.to_datetime(all_features['Listed On'], format="%Y-%m-%d")
all_features['Last Sold On'] = pd.to_datetime(all_features['Last Sold On'], format="%Y-%m-%d")

numeric_features = all_features.dtypes[all_features.dtypes == 'float64'].index
all_features = all_features.fillna(method='bfill', axis=0).fillna(0)
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

features = list(numeric_features)
features.extend(['Type','Bedrooms'])   # 加上类别数相对较少的Type, ,'Cooling features'
all_features = all_features[features]

all_features = pd.get_dummies(all_features,dummy_na=True)

n_train = train_data.shape[0]
f = all_features[:n_train].to_numpy(dtype='float')

train_features = torch.tensor(f, dtype=torch.float32)
print(train_features.dtype)

train_labels = torch.tensor(train_data['Sold Price'].values.reshape(-1, 1), dtype=torch.float32)
print(train_labels.dtype)

# def to_numeric(string):
#     output = ''.join(c for c in string if c.isdigit())
#     if len(output) == 0:
#         return 0
#     else:
#         return output



# for i in range(len(train_data.iloc[:, 34])):
#     train_data.iloc[i, 34] = int(to_numeric(str(train_data.iloc[i, 34])))
#     train_data.iloc[i, 36] = int(to_numeric(str(train_data.iloc[i, 36])))
#     train_data.iloc[i, -2] = int(to_numeric(str(train_data.iloc[i, -2])))
#     train_data.loc[i,"Bedrooms"] = int(to_numeric(str(train_data.loc[i,"Bedrooms"])))

# train_data[["Bedrooms", "Listed On"]] = train_data[["Bedrooms", "Listed On"]].apply(pd.to_numeric)
# train_data[["Bedrooms", "Listed On", "Last Sold On"]] = train_data[["Bedrooms", "Listed On", "Last Sold On"]].replace(0, np.nan)
# train_data = train_data.drop(train_data.columns[[0,1,3,6,7,8, 16, 17, 20, 23, 26, 27, 28, 29, 30, 31, 38, 40]], axis = 1)
# train_labels = torch.tensor(
#     train_data.iloc[:, 0].values.reshape(-1, 1), dtype=torch.float32)
# numeric_features = train_data.dtypes[train_data.dtypes != 'object'].index
# train_data[numeric_features] = train_data[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# train_data[numeric_features] = train_data[numeric_features].fillna(0)
# train_data = pd.get_dummies(train_data, dummy_na=True)
# # object_feature = train_data.dtypes[train_data.dtypes == 'object'].index


num_features = train_features.shape[1]
print(num_features)
# f = train_data.to_numpy(dtype=float)

# train_features = torch.tensor(
#     f, dtype=torch.float32)





dataset = TensorDataset(train_features, train_labels)

hidden_layer1 = 256
p1 = 0.2
p2 = 0.2
learning_rate = 0.5
weight_decay = 0.02


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)



net = torch.nn.Sequential(nn.Linear(num_features, hidden_layer1), nn.ReLU(), nn.Linear(hidden_layer1, 64), \
                          nn.Tanh(), nn.Linear(64, 1))

loss = nn.MSELoss()
updater = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)

net.apply(init_weights)

def log_rmse(net, features, labels, loss_func):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss_func(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

k, num_epochs, lr, weight_decay, batch_size = 5, 2000, 0.005, 0.005, 256
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size, net, loss, log_rmse)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')