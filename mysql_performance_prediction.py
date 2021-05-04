#!/usr/bin/env python
# coding: utf-8

# In[30]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
import pickle
import numpy as np
from glob import glob
from pandas import DataFrame
from pprint import pprint


# In[5]:


all_pkls = glob('../auto_config/configs/mysql_random_config/*.pkl')
result_pkls = list(filter(lambda x: 'result' in x, all_pkls))


# In[7]:


pkl = result_pkls[0]


# In[209]:


len(config)


# In[36]:


df = DataFrame()
data = []
for pkl in result_pkls:
    with open(pkl, 'rb') as f:
        a = pickle.load(f)

    status_set = set()
    for q_id, q_result in a['raw_times'].items():
        status_set.add(q_result['status'])
    if len(status_set) != 1 or len(a['raw_times']) < 22:
        continue
        
#     pprint(a)
#     break
    
    df = df.append({'run_time':a['run_time'],
                   }, ignore_index=True)
    
    # featurize
    config = a['config']
    flush_method = config.pop('innodb_flush_method', None)
    innodb_flush_methods = ["fsync", "O_DSYNC", "littlesync", "nosync", "O_DIRECT", "O_DIRECT_NO_FSYNC",]
    method_index = innodb_flush_methods.index(flush_method)
        
    x = np.zeros(len(config) + len(innodb_flush_methods) )
    
    for i, k in enumerate(sorted(config.keys())):
        x[i] = config[k]
    
    # one-hot encoding
    x[len(config) + method_index] = 1.
    
    y = np.zeros(23)  # 22 single query result + total benchmark
    for q_result in a['raw_times'].values():
        query_idx = q_result['q_type']  # 1-22
        y[query_idx] = q_result['query_run_time']
    
    y[0] = a['run_time']
    
    data.append((x, y))


# In[65]:


len(data[0][0])


# In[67]:


Xs, Ys = np.zeros(( len(data), len(data[0][0])) ), np.zeros((len(data), 23))
for i, (x, y) in enumerate(data):
    Xs[i, :] = x
    Ys[i, :] = y
data = (Xs, Ys)


# In[24]:


plt.plot(df)


# In[124]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso

import GPy
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data


# In[91]:


def split_data(data, data_size):
    Xs, Ys = data
    
    # train, valid, test = [], [], []
    train = Xs[:data_size], Ys[:data_size]
    valid = Xs[data_size:int(data_size*1.1)], Ys[data_size:int(data_size*1.1)]
    test = Xs[-200:], Ys[-200:]
    return train, valid, test


# In[201]:


def shuffle_data(data):
    Xs, Ys = data
    n_samples, _ = Xs.shape
    a = np.arange(n_samples)
    random.shuffle(a)
    new_Xs = Xs[a, :]
    new_Ys = Ys[a, :]
    return (new_Xs, new_Ys)


# In[205]:


data_sizes = [30, 50, 100, 200, 300, 400, 500, 600, 700, 800]
df = DataFrame()

for trial in range(5):
    data = shuffle_data(data)
    for data_size in data_sizes:
        train, valid, test = split_data(data, data_size)

        model_errors = {'data_size': data_size, 'trial':trial}

        model_errors['linear'] = fit_linear_model(train, valid, test)
        model_errors['gpr'] = fit_gpr_model(train, valid, test)
        model_errors['nn'] = fit_nn_model(train, valid, test)
        for alpha in [0., 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 0.8, 1., 2., 4., 5., 8., 10.]:
            model_errors[f'nn_multi alpha={alpha}'] = fit_nn_multi_model(train, valid, test, alpha)

        print(model_errors)
        df = df.append(model_errors, ignore_index=True)
df


# In[206]:


df.to_excel('./model_results.xlsx')


# In[202]:


def fit_nn_multi_model(train, valid, test, alpha=0.5):
    # use all query results
    
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test
    
    sc, sc_target = StandardScaler(), StandardScaler()
    
    train_x, train_y = sc.fit_transform(train_x), sc_target.fit_transform(train_y)
    valid_x, test_x = sc.transform(valid_x), sc.transform(test_x)

    train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
    valid_x, test_x = torch.Tensor(valid_x), torch.Tensor(test_x)
    valid_x, test_x = Variable(valid_x), Variable(test_x)
    
    target_mean, target_scale = torch.Tensor([sc_target.mean_]), torch.Tensor([sc_target.scale_])
    #print(f"mean {target_mean}, scale {target_scale}")
    n_samples, n_features = test_x.shape
    n_samples, n_outputs = test_y.shape
    
    # another way to define a network
    net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, n_outputs),
        )

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    BATCH_SIZE = 64
    EPOCH = 500

    torch_dataset = Data.TensorDataset(train_x, train_y)

    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2,)

    # start training
    min_test_error, min_valid_error = 1e8, 1e8
    patience = 0
    for epoch in range(EPOCH):
        train_loss = 0.
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            b_x, b_y = Variable(batch_x), Variable(batch_y)
            
            normalized_prediction = net(b_x)     # input x and predict based on x
            loss = loss_func(normalized_prediction, b_y)     # must be (1. nn output, 2. target)
            
            prediction = (target_scale * normalized_prediction) + target_mean
            loss_aug = loss_func(prediction[:, 0], torch.sum(prediction[:, 1:], dim=1))
            tmp_loss = loss + alpha * loss_aug
            
            train_loss += tmp_loss
            train_error = torch.mean((prediction - b_y)**2.)
                                 
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            
            print(f"train_error: {train_error:.2f} @ step {step} / epoch {epoch}", end='\r')

        predicted_y = net(valid_x)
        predicted_y = sc_target.inverse_transform(predicted_y.detach())
        valid_error = np.mean((predicted_y[:, 0] - valid_y[:, 0])**2.)
    
        predicted_y = net(test_x)
        predicted_y = sc_target.inverse_transform(predicted_y.detach())
        test_error = np.mean((predicted_y[:, 0] - test_y[:, 0])**2.)
        
        print(f"valid error: {valid_error:.2f} / test error: {test_error:.2f} / epoch {epoch}", end='\r')
        
        if min_valid_error > valid_error:
            min_valid_error = valid_error
            min_test_error = test_error
            patience = 0
        else:
            patience += 1
        
        if patience == 10:
            break
    return min_test_error


# In[203]:


def fit_nn_model(train, valid, test):
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test
    
    train_y = train_y[:, 0].reshape(-1, 1)
    valid_y = valid_y[:, 0].reshape(-1, 1)
    test_y = test_y[:, 0].reshape(-1, 1)
    
    sc, sc_target = StandardScaler(), StandardScaler()
    
    train_x, train_y = sc.fit_transform(train_x), sc_target.fit_transform(train_y).reshape(-1, 1)
    valid_x, test_x = sc.transform(valid_x), sc.transform(test_x)

#     valid_x, valid_y = sc.transform(valid_x), sc_target.transform(valid_y).reshape(-1, 1)
#     test_x, test_y = sc.transform(test_x), sc_target.transform(test_y).reshape(-1, 1)

    train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
    valid_x, test_x = torch.Tensor(valid_x), torch.Tensor(test_x)
    valid_x, test_x = Variable(valid_x), Variable(test_x)

    n_samples, n_features = test_x.shape
    
    # another way to define a network
    net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1),
        )

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    BATCH_SIZE = 64
    EPOCH = 500

    torch_dataset = Data.TensorDataset(train_x, train_y)

    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2,)

    # start training
    min_test_error, min_valid_error = 1e8, 1e8
    patience = 0
    for epoch in range(EPOCH):
        train_loss = 0.
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            b_x, b_y = Variable(batch_x), Variable(batch_y)
            
            prediction = net(b_x)     # input x and predict based on x

            loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
            train_loss += loss
            
            train_error = torch.mean((prediction - b_y)**2.)
            
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            
            print(f"train_error: {train_error:.2f} @ step {step} / epoch {epoch}", end='\r')

        predicted_y = net(valid_x)
        predicted_y = sc_target.inverse_transform(predicted_y.detach())
        valid_error = np.mean((predicted_y - valid_y)**2.)
    
        predicted_y = net(test_x)
        predicted_y = sc_target.inverse_transform(predicted_y.detach())
        test_error = np.mean((predicted_y - test_y)**2.)
        
        print(f"valid error: {valid_error:.2f} / test error: {test_error:.2f} / epoch {epoch}", end='\r')
        
        if min_valid_error > valid_error:
            min_valid_error = valid_error
            min_test_error = test_error
            patience = 0
        else:
            patience += 1
        
        if patience == 10:
            break
    return min_test_error


# In[204]:


def fit_gpr_model(train, valid, test):
    Xs, Ys = train
    Xs_2, Ys_2 = valid
    
    Xs = np.concatenate([Xs, Xs_2], axis=0)
    Ys = np.concatenate([Ys, Ys_2], axis=0)
    Ys = Ys[:, 0].reshape(-1, 1)
    
    n_samples, n_features = Xs.shape
    
    sc, sc_target = StandardScaler(), StandardScaler()  # [n_samples, n_features]

    normalized_Xs = sc.fit_transform(Xs)
    normalized_Ys = sc_target.fit_transform(Ys[:, 0].reshape(-1, 1)).reshape(-1, 1)

    kernel = GPy.kern.RBF(input_dim=n_features)
    m = GPy.models.GPRegression(normalized_Xs, normalized_Ys, kernel=kernel)
    m.optimize('bfgs', max_iters=200)
    
    test_Xs, test_Ys = test
    test_Ys = test_Ys[:, 0].reshape(-1, 1)
    normalized_Xs = sc.transform(test_Xs)
    
    predicted_normalized_Ys, _ = m.predict(normalized_Xs)
    predicted_Ys = sc_target.inverse_transform(predicted_normalized_Ys)    
    return np.mean(np.abs(test_Ys - predicted_Ys)**2.)

def fit_linear_model(train, valid, test):
    Xs, Ys = train
    Xs_2, Ys_2 = valid
    
    Xs = np.concatenate([Xs, Xs_2], axis=0)
    Ys = np.concatenate([Ys, Ys_2], axis=0)
    Ys = Ys[:, 0].reshape(-1, 1)
#     print(Xs.shape, Ys.shape)
    
    model = Lasso()
    results = model.fit(Xs, Ys)
    
    Xs, Ys = test
    Ys = Ys[:, 0]
    predicted_Ys = model.predict(Xs)
    return np.mean(np.abs(Ys - predicted_Ys)**2.)


# In[ ]:




