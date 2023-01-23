import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Standard loss functions

def mse(preds, targets):
  loss = nn.MSELoss()
  return loss(preds, targets)

def mae(preds, targets):
  loss = nn.L1Loss()
  return loss(preds, targets)

def smooth_mae(preds, targets):
  loss = nn.SmoothL1Loss()
  return loss(preds, targets)

def huber(preds, targets, delta=1.):
  loss = nn.HuberLoss()
  return loss(preds, targets)

# Weighted loss functions

def weighted_mse(preds, targets, weights=5.):
    loss = (preds - targets) ** 2
    if weights is not None:
        #loss *= weights.expand_as(loss)
        loss *= torch.tensor(weights).expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_mae(preds, targets, weights=5.):
    loss = F.l1_loss(preds, targets, reduction='none')
    if weights is not None:
        loss *= torch.tensor(weights).expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_huber(preds, targets, weights=5., beta=1.):
    l1_loss = torch.abs(preds - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= torch.tensor(weights).expand_as(loss)
    loss = torch.mean(loss)
    return loss

# Focal-R loss functions (Yang et al., 2021)

def weighted_focal_mse(preds, targets, weights=5., activate='sigmoid', beta=.2, gamma=1.):
    loss = (preds - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(preds - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(preds - targets)) - 1) ** gamma
    if weights is not None:
        loss *= torch.tensor(weights).expand_as(loss)
    loss = torch.mean(loss)
    return loss
    
def weighted_focal_mae(preds, targets, weights=5., activate='sigmoid', beta=.2, gamma=1.):
    loss = F.l1_loss(preds, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(preds - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(preds - targets)) - 1) ** gamma
    if weights is not None:
        loss *= torch.tensor(weights).expand_as(loss)
    loss = torch.mean(loss)
    return loss

# Balanced MSE loss function (Ren et al., 2022)

def balanced_mse(preds, targets, noise_var):
    logits = - (preds - targets.T).pow(2) / (2 * noise_var)
    loss = F.cross_entropy(logits, torch.arange(preds.shape[0]))
    loss = loss * (2 * noise_var)
    return loss

# Customized weighted loss functions

def cm_weighted_mae(preds, targets, threshold=1., weight=5.):
    weights = []
    for i in targets:
        if i >= threshold:
            weights.append(weight)
        else:
            weights.append(1)
    weights = Variable(torch.Tensor(weights))
    unweighted_loss = abs(preds - targets)
    loss = unweighted_loss * weights.expand_as(weights)
    loss = loss.mean() 
    return loss