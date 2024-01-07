# @Time    : 2023/9/12  17:35
# @Auther  : Teng Zhang
# @File    : Proposed.py
# @Project : Energy
# @Software: PyCharm



import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import Result_evalute
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.utils.data as Data
import os
from sklearn.model_selection import train_test_split
import time
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams["font.size"] = 10
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.FC1 = nn.Linear(8, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC3 = nn.Linear(64, 32)
        self.predict = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.FC1(x)
        x = self.relu(x)
        x = self.FC2(x)
        x = self.relu(x)
        x = self.FC3(x)
        x = self.relu(x)
        result = self.predict(x)
        return result


def qd_objective(y_true, y_pred):

    '''Loss_QD-soft, from algorithm 1'''
    shape = np.shape(y_true)
    num_dimension = int(shape[1] / 2)
    Loss_S_sum = 0
    for dimension in range(num_dimension):
        y_true_dimension = y_true[:, dimension * 2]
        y_u = y_pred[:, dimension * 2]
        y_l = y_pred[:, dimension * 2 + 1]
        zero_tensor = torch.tensor(0).to(DEVICE)
        K_HU = torch.max(zero_tensor, torch.sigmoid(y_u - y_true_dimension))
        K_HL = torch.max(zero_tensor, torch.sigmoid(y_true_dimension - y_l))
        K_H = torch.mul(K_HU, K_HL)

        K_SU = torch.sigmoid(soften_ * (y_u - y_true_dimension))
        K_SL = torch.sigmoid(soften_ * (y_true_dimension - y_l))
        K_S = torch.mul(K_SU, K_SL)

        MPIW_c = torch.sum(torch.mul((y_u - y_l), K_H)) / (torch.sum(K_H) + 0.001)
        PICP_H = torch.mean(K_H)
        PICP_S = torch.mean(K_S)

        Loss_S = MPIW_c + lambda_ * Batch_size / (alpha_ * (1 - alpha_)) * torch.square(
            torch.max(zero_tensor, (1 - alpha_) - PICP_S))
        Loss_S_sum = Loss_S_sum + Loss_S
    return Loss_S_sum


def split_sequences(inputs, output, length, forward, gap):
    # 数据末尾不重合，并且可修改滑窗步距
    X, y = list(), list()
    for i in range(len(inputs)):
        index = i * gap
        start = index
        stop = index + length
        stop_forward = stop + forward
        if stop_forward > len(inputs) - forward: break
        seq_x, seq_y = inputs[start: stop], output[stop:stop_forward, :]
        X.append(seq_x), y.append(seq_y)
    y = np.squeeze(np.array(y))
    return np.array(X), np.array(y)


def ParametersTransfer(model_new, model_old):
    pretrained_dict = model_old.state_dict()
    new_dict = model_new.state_dict()
    new_dict_fc = model_new.state_dict()
    pretrained_dict.popitem(last=True)
    pretrained_dict.popitem(last=True)
    new_dict.popitem(last=True)
    new_dict.popitem(last=True)

    pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in new_dict}
    pretrained_dict1['predict.weight'] = new_dict_fc['predict.weight']
    pretrained_dict1['predict.bias'] = new_dict_fc['predict.bias']
    new_dict.update(pretrained_dict1)
    model_new.load_state_dict(new_dict)
    for _, value in model_new.named_parameters():
        value.requires_grad = True
    print('Moldel loading finished')


if __name__ == '__main__':
    T1 = time.time()
    SaveData = False

    # 数据准备
    data = pd.read_csv('data/Energy.csv', header=0)
    data = np.array(data)
    theta = data[:, 0:8]
    error = data[:, 8]
    error = error.reshape(-1, 1)
    ss = StandardScaler()
    theta = ss.fit_transform(theta)

    shape_label = np.shape(error)
    y_train_P = error
    X_train = theta

    X = Variable(torch.Tensor(X_train)).to(DEVICE)
    y_P = Variable(torch.Tensor(y_train_P)).to(DEVICE)

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y_P, shuffle=True, test_size=0.7)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, shuffle=True, test_size=0.5)

    ''':param
    # explaination:
    X       y_P     : Original_data
    X_train Y_train : general training data
    X_val   Y_val   : Calibration training data
    X_test  Y_test  : testing data
    '''

    numEnsemble = 10
    data_pred_P_sum = Variable(torch.Tensor(np.zeros(np.shape(y_train_P)))).to(DEVICE)
    data_pred_PTrain_sum = Variable(torch.Tensor(np.zeros(np.shape(Y_train)))).to(DEVICE)
    data_pred_PVal_sum = Variable(torch.Tensor(np.zeros(np.shape(Y_val)))).to(DEVICE)
    data_pred_PTest_sum = Variable(torch.Tensor(np.zeros(np.shape(Y_test)))).to(DEVICE)

    data_pred_PI_sum = Variable(torch.Tensor(np.zeros([len(y_train_P), 2 * shape_label[1]]))).to(DEVICE)
    data_pred_PI_Hold = Variable(torch.Tensor(np.zeros([numEnsemble, len(y_train_P), 2 * shape_label[1]]))).to(DEVICE)
    numOK = 0
    while numOK != numEnsemble:
        #  *********************nuclear predictor*********************
        if 1:
            # model structure
            output_size = shape_label[1]
            # model parameters
            learning_rate = 1e-2
            regularization = 1e-3
            epoch_num = 80
            Batch_size = 32

            lstm_P = ANN().to(DEVICE)
            optimizer = optim.Adam(lstm_P.parameters(), lr=learning_rate, weight_decay=regularization)
            loss_fn = nn.MSELoss()
            torch_dataset = Data.TensorDataset(X_train, Y_train)
            loader = Data.DataLoader(dataset=torch_dataset, batch_size=Batch_size, shuffle=True)
            result_loss_P = []
            for epoch in range(epoch_num):
                for step, (batch_x, batch_y) in enumerate(loader):
                    predict_error = lstm_P(batch_x)
                    loss = loss_fn(predict_error, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if epoch % 1 == 0:
                    result_loss_P.append(loss.data.cpu() / len(batch_x))
                if epoch % 10 == 0:
                    print("epoch: {}  loss: {:.5f}".format(str(epoch), loss.item()))
            result_loss_P = np.array(result_loss_P)
            x = range(result_loss_P.shape[0])
            data_pred_P = lstm_P(X)
            data_pred_P_sum = data_pred_P_sum + data_pred_P

        # *********************ancillary predictor*********************
        if 1:

            lambda_ = 0.01
            alpha_ = 0.05
            soften_ = 5.
            output_size = 2 * shape_label[1]
            learning_rate = 1e-2
            regularization = 1e-5
            epoch_num = 200
            Batch_size = 64

            X_train_Sample, X_unUse, Y_train_Sample, Y_val_unUse = train_test_split(X_train, Y_train, shuffle=True, test_size=0.05)
            y_train_PI = np.zeros([len(X_train_Sample), output_size])
            for dimension in range(shape_label[1]):
                y_train_PI[:, 2 * dimension] = Y_train_Sample[:, dimension].data.cpu()
                y_train_PI[:, 2 * dimension + 1] = Y_train_Sample[:, dimension].data.cpu()

            y_PI = Variable(torch.Tensor(y_train_PI)).to(DEVICE)
            lstm_PI = ANN().to(DEVICE)
            result_loss_PI = []
            ParametersTransfer(lstm_PI, lstm_P)
            optimizer = optim.Adam(lstm_PI.parameters(), lr=learning_rate, weight_decay=regularization)
            torch_dataset = Data.TensorDataset(X_train_Sample, y_PI)
            loader = Data.DataLoader(dataset=torch_dataset, batch_size=Batch_size, shuffle=True)
            for epoch in range(epoch_num):
                for step, (batch_x, batch_y) in enumerate(loader):
                    predict_error = lstm_PI(batch_x)
                    loss = qd_objective(batch_y, predict_error)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if epoch % 1 == 0:
                    result_loss_PI.append(loss.data.cpu() / len(batch_x))
                if epoch % 10 == 0:
                    print("epoch: {}  loss: {:.5f}".format(str(epoch), loss.item()))
            result_loss = np.array(result_loss_PI).reshape(-1)
            data_pred_PI = lstm_PI(X)
            result_PI = Result_evalute.PI_index_HD(data_pred_PI.data.cpu(), y_train_P, alpha=0.05, eta=50)
            numOK = numOK + 1

    data_true = y_train_P
    data_pred_P = data_pred_P_sum.data.cpu() / numEnsemble
    data_pred_PI_mean = data_pred_PI_sum.data.cpu() / numEnsemble
    data_pred_PI_std = torch.std(data_pred_PI_Hold, dim=0).data.cpu()
    data_pred_PI_var = torch.var(data_pred_PI_Hold, dim=0).data.cpu()
    model_Uncertainty = Variable(torch.Tensor(np.zeros([len(y_train_P), shape_label[1]]))).to(DEVICE)
    model_Uncertainty_Prepare = Variable(torch.Tensor(np.zeros([len(y_train_P), 2]))).to(DEVICE)
    if numEnsemble == 1:
        model_Uncertainty = Variable(torch.Tensor(np.zeros([len(y_train_P), shape_label[1]]))).to(DEVICE)
    else:
        for dimension in range(shape_label[1]):
            data_pred_PI[:, 2 * dimension] = data_pred_PI_mean[:, 2 * dimension] + 1.96 * data_pred_PI_std[:, 2 * dimension]
            data_pred_PI[:, 2 * dimension + 1] = data_pred_PI_mean[:, 2 * dimension + 1] - 1.96 * data_pred_PI_std[:, 2 * dimension + 1]
            model_Uncertainty_Prepare[:, 0] = data_pred_PI_var[:, 2 * dimension]
            model_Uncertainty_Prepare[:, 1] = data_pred_PI_var[:, 2 * dimension + 1]
            model_Uncertainty[:, dimension] = torch.mean(model_Uncertainty_Prepare, dim=1)

    datamiddle = data_pred_PI.clone()
    data_pred_PI[:, 0] = datamiddle[:, 1]
    data_pred_PI[:, 1] = datamiddle[:, 0]

    if shape_label[1] == 6:
        Result_evalute.predict_positon(data_true, data_pred_P.data.cpu())
        Result_evalute.predict_orientation(data_true, data_pred_P.data.cpu())
        Result_evalute.PI_index_position(data_pred_PI.data.cpu(), data_true, alpha=0.05, eta=50)
        Result_evalute.PI_index_orientation(data_pred_PI.data.cpu(), data_true, alpha=0.05, eta=50)
    else:
        Result_evalute.predict(data_true, data_pred_P.data.cpu())
        Result_evalute.PI_index_HD(data_pred_PI.data.cpu(), data_true, alpha=0.05, eta=50)


    if SaveData:
        Y_all_pre = data_pred_P.data.cpu().numpy()
        path_all_P = 'P_All_P.csv'
        principle = pd.DataFrame(data=Y_all_pre)
        principle.to_csv(path_all_P)

        # 区间预测值
        path_all_PI = 'PI_All.csv'
        principle = pd.DataFrame(data=data_pred_PI.data.cpu().numpy())
        principle.to_csv(path_all_PI)

    T2 = time.time()
    print('All time is :%s秒' % ((T2 - T1)))
    plt.show()