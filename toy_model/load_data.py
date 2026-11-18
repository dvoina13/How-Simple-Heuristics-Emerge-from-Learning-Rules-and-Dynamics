import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import random
import pickle
from sklearn.metrics import roc_auc_score

from PIL import Image
from scipy.signal import fftconvolve
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import argparse

def load_data(n_neurons = 100, 
              n_samples_train=1000, 
              n_samples_test=1000, 
              p_over = 0.8, 

              phase = "1", 
              gain_to_add = 0.0,
              
              amplitude_or = 5.0, 
              amplitude_ur = 3.0, 
              baseline = 26.0,
              amplitude_untuned = 10.0,
              
              noise_or=0.1, 
              noise_ur = 0.1, 
              noise_bg = 0.1, 
              noise_independent = 0.1, 

              use_noise_model = True,
              multiplicative_noise = False,
              poisson_noise = False,
              lognormal_noise = False,
              untuned_neurons = False,

              centering = False,
              seed = 1,
              
              batchsize=1):

    rng = np.random.default_rng(seed)

    if phase == "2":
        amplitude_ur = amplitude_or
        
    X_train, y_train, pref_train = generate_dataset(n_neurons, n_samples_train, p_over, amplitude_or, amplitude_ur, baseline, amplitude_untuned, noise_or, noise_ur, noise_bg, noise_independent, use_noise_model, multiplicative_noise, poisson_noise, lognormal_noise, untuned_neurons, seed1 = seed, seed2 = seed)
    X_test, y_test, pref_test = generate_dataset(n_neurons, n_samples_test, p_over, amplitude_or, amplitude_ur, baseline, amplitude_untuned, noise_or, noise_ur, noise_bg, noise_independent, use_noise_model, multiplicative_noise, untuned_neurons, seed1 = seed, seed2 = 100+seed)

    #center or not data
    if centering:
        mean = X_train.mean(0).copy()
        X_train = X_train - mean
        X_test = X_test - mean

    X_train += gain_to_add
    X_test += gain_to_add

    g = torch.Generator().manual_seed(seed)
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=n_samples_test, shuffle=False)

    return train_dataloader, test_dataloader, X_train, y_train, X_test, y_test, (pref_train, pref_test)

def generate_dataset(
    N=100, 
    n_samples=1000, 
    p_over=0.8,
    amplitude_or=30.0, 
    amplitude_ur=10.0, 
    baseline=10.0, 
    amplitude_untuned=10.0,
    
    noise_or=5.0, 
    noise_ur=5.0, 
    noise_bg=0.5, 
    noise_independent=1.0,
    
    use_noise_model=True, 
    multiplicative_noise=False, 
    poisson_noise=False,
    lognormal_noise=False,
    untuned_neurons=False, 
    
    p_untuned=0.2, 
    noise_untuned=0.1,
    spikes=False, 
    dt=1.0, 
    seed1=0,
    seed2=0,
    n_trials_per_condition=0.5,
):
    
    rng = np.random.default_rng(seed1)
    n_trials = int(n_trials_per_condition * n_samples)

    p_u = p_untuned if untuned_neurons else 0.0
    pref_array = rng.choice([0, 1, 2], size=N,
                            p=[p_u, (1 - p_u) * p_over, (1 - p_u) * (1 - p_over)])

    means_g1 = np.array([amplitude_untuned, baseline + amplitude_or, baseline]) * dt
    means_g2 = np.array([amplitude_untuned, baseline, baseline + amplitude_ur]) * dt
    r1, r2 = means_g1[pref_array], means_g2[pref_array]
    
    bg2 = noise_bg**2
    noise_cov = np.array([[noise_untuned**2 + bg2, bg2, bg2],
                          [bg2, noise_or**2 + bg2, bg2],
                          [bg2, bg2, noise_ur**2 + bg2]])
    C = noise_cov[pref_array[:, None], pref_array[None, :]]

    # ---- Gaussian path: total covariance + Cholesky ----
    def total_cov(r):
        shared = C * np.outer(r, r) if multiplicative_noise else C
        private = np.diag(np.maximum(r, 1e-6)) if poisson_noise else noise_independent**2 * np.eye(N)
        return shared + private if use_noise_model else private

    def sample_gaussian(r):
        L = np.linalg.cholesky(total_cov(r))
        X = r + rng.standard_normal((n_trials, N)) @ L.T
        if spikes:   # set poisson_noise=False to avoid double-counting
            X = rng.poisson(np.clip(X, 0, None))
        return X

    # ---- Lognormal path: sample latent gain, then private noise ----
    def sample_lognormal(r):
        if use_noise_model:
            z = rng.multivariate_normal(np.zeros(3), noise_cov, size=n_trials)  # (n_trials, 3)
            eta = z[:, pref_array]                                              # (n_trials, N)
            gain = np.exp(eta - 0.5 * np.diag(noise_cov)[pref_array])           # E[gain] = 1
        else:
            gain = np.ones((n_trials, N))
        lam = r * gain
        if poisson_noise or spikes:   # exact Poisson, valid at any count
            return rng.poisson(lam)
        return lam + noise_independent * rng.standard_normal((n_trials, N))

    rng = np.random.default_rng(seed2)
    sample = sample_lognormal if lognormal_noise else sample_gaussian
    X = np.r_[sample(r1), sample(r2)]
    y = np.r_[np.ones(n_trials), np.zeros(n_trials)]

    if (X<0).sum() > 0:
        raise ValueError("Firing rate cannot be negative.")
        
    return X, y, pref_array


