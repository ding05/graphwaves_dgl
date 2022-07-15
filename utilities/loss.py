import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard loss functions

def mse(preds, target):
  return nn.MSELoss(preds, targets)

def mae(preds, targets):
  return nn.L1Loss(preds, targets)

def smooth_mae(preds, targets):
  return nn.SmoothL1Loss(preds, targets)

def huber(preds, targets, delta=1.):
  return nn.HuberLoss(preds, targets, delta)

# Weighted loss functions

def weighted_mse(preds, targets, weights=None):
    loss = (preds - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_mae(preds, targets, weights=None):
    loss = F.l1_loss(preds, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_huber_loss(preds, targets, weights=None, beta=1.):
    l1_loss = torch.abs(preds - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

# Focal-R loss functions (Yang et al., 2021)

def weighted_focal_mse(preds, targets, weights=None, activate='sigmoid', beta=.2, gamma=1.):
    loss = (preds - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(preds - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(preds - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
    
def weighted_focal_mae(preds, targets, weights=None, activate='sigmoid', beta=.2, gamma=1.):
    loss = F.l1_loss(preds, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(preds - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(preds - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

# Balanced MSE loss function (Ren et al., 2022)

def balanced_mse(preds, targets, noise_var=1.):
    logits = - (preds - targets.T).pow(2) / (2 * noise_var)
    loss = F.cross_entropy(logits, torch.arange(preds.shape[0]))
    loss = loss * (2 * noise_var)
    return loss