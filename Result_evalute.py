# -*- coding: utf-8 -*-
"""
Created on 14:11,2023/09/13
@author: ZhangTeng
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import math


def to_numpy(lowers, uppers, true):
    if torch.is_tensor(lowers):
        lowers = lowers.numpy().squeeze()
    if torch.is_tensor(uppers):
        uppers = uppers.numpy().squeeze()
    if torch.is_tensor(true):
        true = true.numpy().squeeze()

    return lowers, uppers, true


def predict(y_true, y_pred):
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    print('****** Points Prediction ******')
    print("MAE  {}".format(mean_absolute_error(y_true, y_pred)))
    print("RMSE  {}".format(np.sqrt(mean_squared_error(y_true, y_pred))))

    result = np.array([MAE, RMSE])
    return result


def predict_positon(y_t, y_p):
    """
    y_t: 实际标签
    y_p: 预测标签
    return: 返回四个评价指标所组成的array
    """
    y_true = y_t[:, 0:3]
    y_pred = y_p[:, 0:3]

    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    print('****** Points Prediction positon error ******')
    print("MAE  {}".format(mean_absolute_error(y_true, y_pred)))
    print("RMSE  {}".format(np.sqrt(mean_squared_error(y_true, y_pred))))
    result = np.array([MAE, RMSE])
    return result


def predict_orientation(y_t, y_p):
    """
    y_t: 实际标签
    y_p: 预测标签
    return: 返回四个评价指标所组成的array
    """
    y_true = y_t[:, 3:6]
    y_pred = y_p[:, 3:6]

    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    print('****** Points Prediction orientation error******')
    print("MAE  {}".format(mean_absolute_error(y_true, y_pred)))
    print("RMSE  {}".format(np.sqrt(mean_squared_error(y_true, y_pred))))
    result = np.array([MAE, RMSE])
    return result



def PI_index_1D(lowers, uppers, true, alpha=0.05, eta=50):

    mu = 1 - alpha
    coverage = 0
    MPIW = 0
    for i in range(len(true)):
        MPIW = MPIW + (uppers[i] - lowers[i])
        if lowers[i] <= true[i] <= uppers[i]:
            coverage = coverage + 1

    PICP = coverage / len(true)
    MPIW = MPIW / len(true)
    PINAW = MPIW / (np.max(true) - np.min(true))
    gamma = 1 if PICP < mu else 0
    CWC = PINAW * (1 + gamma * math.exp(-eta * (PICP - mu)))
    print('****** Uncertainties estimation ******')
    print("PICP:", PICP)
    print("MPIW:", np.array(MPIW))
    print("PINAW: ", np.array(PINAW))
    result = np.array([PICP, MPIW, PINAW])
    return result


def PI_index_HD(PI, label, alpha=0.05, eta=50):
    mu = 1 - alpha
    dimension_label = label.shape[1]
    PICP_sum = 0
    MPIW_sum = 0
    PINAW_sum = 0
    CWC_sum = 0

    for dd in range(dimension_label):
        # 这两个必须在内训练中开展
        coverage = 0
        MPIW = 0
        uppers = PI[:, 2 * dd].cpu().data.numpy()
        lowers = PI[:, 2 * dd + 1].cpu().data.numpy()
        true = label[:, dd]
        for i in range(len(true)):
            MPIW = MPIW + (uppers[i] - lowers[i])
            if lowers[i] <= true[i] <= uppers[i]:
                coverage = coverage + 1

        PICP = coverage / len(true)
        PICP_sum = PICP_sum + PICP
        MPIW = MPIW / len(true)
        MPIW_sum = MPIW_sum + MPIW
        PINAW = MPIW / (np.max(true) - np.min(true))
        PINAW_sum = PINAW_sum + PINAW
        gamma = 1 if PICP < mu else 0
        CWC = PINAW * (1 + gamma * math.exp(-eta * (PICP - mu)))
        CWC_sum = CWC_sum + CWC

    print('****** Uncertainties estimation ******')
    print("PICP:", PICP_sum / dimension_label)
    print("MPIW:", MPIW_sum / dimension_label)
    print("PINAW: ", PINAW_sum / dimension_label)
    result = np.array([PICP_sum, MPIW_sum, PINAW_sum])
    return result


def PI_index_position(PI, label, alpha=0.05, eta=50):

    mu = 1 - alpha

    dimension_label = int(label.shape[1]/2)
    PICP_sum = 0
    MPIW_sum = 0
    PINAW_sum = 0
    CWC_sum = 0
    PI = PI[:,0:6]
    label = label[:, 0:3]
    for dd in range(dimension_label):
        # 这两个必须在内训练中开展
        coverage = 0
        MPIW = 0
        uppers = PI[:, 2 * dd].cpu().data.numpy()
        lowers = PI[:, 2 * dd + 1].cpu().data.numpy()
        true = label[:, dd]
        for i in range(len(true)):
            MPIW = MPIW + (uppers[i] - lowers[i])
            if lowers[i] <= true[i] <= uppers[i]:
                coverage = coverage + 1

        PICP = coverage / len(true)
        PICP_sum = PICP_sum + PICP
        MPIW = MPIW / len(true)
        MPIW_sum = MPIW_sum + MPIW
        PINAW = MPIW / (np.max(true) - np.min(true))
        PINAW_sum = PINAW_sum + PINAW
        gamma = 1 if PICP < mu else 0
        CWC = PINAW * (1 + gamma * math.exp(-eta * (PICP - mu)))
        CWC_sum = CWC_sum + CWC

    print('****** Uncertainties estimation of Position error******')
    print("PICP:", PICP_sum / dimension_label)
    print("MPIW:", MPIW_sum / dimension_label)
    print("PINAW: ", PINAW_sum / dimension_label)
    result = np.array([PICP_sum, MPIW_sum, PINAW_sum])
    return result


def PI_index_orientation(PI, label, alpha=0.05, eta=50):
    mu = 1 - alpha
    dimension_label = int(label.shape[1]/2)
    PICP_sum = 0
    MPIW_sum = 0
    PINAW_sum = 0
    CWC_sum = 0
    PI = PI[:,6:12]
    label = label[:, 3:6]
    for dd in range(dimension_label):
        # 这两个必须在内训练中开展
        coverage = 0
        MPIW = 0
        uppers = PI[:, 2 * dd].cpu().data.numpy()
        lowers = PI[:, 2 * dd + 1].cpu().data.numpy()
        true = label[:, dd]
        for i in range(len(true)):
            MPIW = MPIW + (uppers[i] - lowers[i])
            if lowers[i] <= true[i] <= uppers[i]:
                coverage = coverage + 1

        PICP = coverage / len(true)
        PICP_sum = PICP_sum + PICP
        MPIW = MPIW / len(true)
        MPIW_sum = MPIW_sum + MPIW
        PINAW = MPIW / (np.max(true) - np.min(true))
        PINAW_sum = PINAW_sum + PINAW
        gamma = 1 if PICP < mu else 0
        CWC = PINAW * (1 + gamma * math.exp(-eta * (PICP - mu)))
        CWC_sum = CWC_sum + CWC

    print('****** Uncertainties estimation of orientaiton error ******')
    print("PICP:", PICP_sum / dimension_label)
    print("MPIW:", MPIW_sum / dimension_label)
    print("PINAW: ", PINAW_sum / dimension_label)
    result = np.array([PICP_sum, MPIW_sum, PINAW_sum])
    return result