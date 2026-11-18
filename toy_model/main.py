#teacher-student on toy data; teacher backprop/GD; student learning rules
import numpy as np
import torch
import random
import os

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

from PIL import Image
from scipy.signal import fftconvolve
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from load_data import load_data
from train import train_model, test_model_simple, train_Hebbian, train_oja, train_oja_unsupervised, train_oja_unsupervised_, train_oja_with_bias_unsupervised, train_3factor_hebbian
from utils import orient_weights, find_tuning, plot_weights, find_bias_simple, test_learning_rule, compute_cp, compute_noise_corr, compute_readout_weights, compute_d_prime, compute_dprime_cp_correlation

import parameters
from parameters import *

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = 'cpu'

if p_over == 0.5:
    phase = '2'
    
#load data
train_dataloader, test_dataloader, X_train, y_train, X_test, y_test, tuning_groups = load_data(n_neurons = n_neurons, p_over = p_over, phase=phase, gain_to_add = gain_to_add, noise_or=noise_group, noise_ur = noise_group, noise_bg = noise_common, noise_independent = noise_independent, centering = centering, seed=seed, batchsize = batchsize)

#find tunings of neurons
neurons_tuned_stim0, neurons_tuned_stim1, neurons_untuned = find_tuning(torch.tensor(X_train).squeeze(), torch.tensor(y_train), min_abs_dprime=0.2, seed=seed); tunings = [neurons_tuned_stim0, neurons_tuned_stim1, neurons_untuned]

#load model
teacher_network = nn.Linear(n_neurons, 1)
if crit == 'MSE': 
    PATH = "mlp_head_MSE.pth"
    criterion = nn.MSELoss() #nn.BCEWithLogitsLoss()

else:
    PATH = "mlp_head_BCE.pth"
    criterion = nn.BCEWithLogitsLoss()
    
if optimize == 'Adam':
    optimizer = optim.Adam(teacher_network.parameters(), lr=0.01)
elif optimize == 'SGD':
    optimizer = optim.SGD(teacher_network.parameters(), lr=0.01)

parameter_list = {}
parameter_list["criterion"] = crit; parameter_list["p_over"] = p_over; parameter_list["W_init"] = "no_initW" #"initW1"
parameter_list["noise_gp_co_ind"] = [noise_group, noise_common, noise_independent]; parameter_list["seed"] = seed; parameter_list["phase"] = phase; parameter_list["centering"] = "_centering" if centering else "_no_centering"

teacher_network, training_acc, valid_acc = train_model(teacher_network, train_dataloader, test_dataloader, optimizer, criterion, n_epochs, parameter_list, phase, device)
weights_GD = teacher_network.weight; bias_GD = teacher_network.bias

torch.save(teacher_network.state_dict(), PATH)
#teacher_network.load_state_dict(torch.load(PATH))

valid_acc, class_stim0, class_stim1 = test_learning_rule(X_test, y_test, weights_GD, bias_GD, crit = crit)

acc_lesioning_GD = []; class_stim0_arr_GD = []; class_stim1_arr_GD = []
p_arr = [0.0001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 0.9]
for p in p_arr:
    val_acc, class_stim0, class_stim1 = find_bias_simple(p, weights_GD, bias_GD, torch.tensor(X_test), torch.tensor(y_test), crit, seed=seed)
    acc_lesioning_GD.append(val_acc); class_stim0_arr_GD.append(class_stim0); class_stim1_arr_GD.append(class_stim1)

cp_GD, CP_per_stim, neurons_tuned_list, prefs, choice = compute_cp(weights_GD, bias_GD, torch.tensor(X_test).float(), torch.tensor(y_test), tunings, crit)
readout_weights_GD, w_raw, neurons, explained, cp_rw_GD, rho_GD = compute_readout_weights(weights_GD, bias_GD, torch.tensor(X_test).float(), torch.tensor(y_test), tunings, crit)

additional = "_no_centering" if centering == False else "_centering"
file_ending = "_crit_" + crit + "_p_over_" + str(p_over) + "_"+ parameter_list["W_init"] + "_noise_gp_co_ind_" + str(noise_group) + "_" + str(noise_common) + "_" + str(noise_independent) +  "_seed_" + str(seed) + "_phase" + phase + additional + ".pt"

torch.save(weights_GD, "results/GD_new_model/weights_GD" + file_ending)
torch.save(bias_GD, "results/GD_new_model/bias_GD" + file_ending)
torch.save(valid_acc, "results/GD_new_model/val_acc_GD" + file_ending)
torch.save(acc_lesioning_GD, "results/GD_new_model/acc_lesioning_GD" + file_ending)
torch.save(class_stim0_arr_GD, "results/GD_new_model/class_stim0_arr_GD" + file_ending)
torch.save(class_stim1_arr_GD, "results/GD_new_model/class_stim1_arr_GD" + file_ending)
torch.save(cp_GD, "results/GD_new_model/cp_GD" + file_ending)
torch.save(readout_weights_GD, "results/GD_new_model/readout_weights_GD" + file_ending)
torch.save(tunings[0], "results/GD_new_model/tuning_stim0_GD" + file_ending)
torch.save(tunings[1], "results/GD_new_model/tuning_stim1_GD" + file_ending)

######
train_dataloader, test_dataloader, X_train, y_train, X_test, y_test, tuning_groups = load_data(n_neurons = n_neurons, p_over = p_over, phase=phase, gain_to_add = gain_to_add, noise_or=noise_group, noise_ur = noise_group, noise_bg = noise_common, noise_independent = noise_independent, centering = centering, seed=seed)

#find tunings of neurons
neurons_tuned_stim0, neurons_tuned_stim1, neurons_untuned = find_tuning(torch.tensor(X_train).squeeze(), torch.tensor(y_train), min_abs_dprime=0.2, seed=seed); tunings = [neurons_tuned_stim0, neurons_tuned_stim1, neurons_untuned]

experiment_name = "Oja_unsupervised_onesW"; extra_name = "_seed" + str(seed); #"_different_init" #""
print("This is an " + experiment_name + " experiment")
crit = "BCE"

if "Hebbian" in experiment_name:
    weights, bias, eta = train_Hebbian(teacher_network, X_train)
elif "Oja_with_teacher" in experiment_name:
    weights, bias, eta = train_oja(teacher_network, X_train)
elif "Oja_unsupervised" in experiment_name:
    weights, bias, eta, acc_arr = train_oja_unsupervised_(teacher_network, X_train, y_train, phase=phase, parameter_list = parameter_list)
elif "Oja_with_bias_unsupervised" in experiment_name:
    weights, bias, eta = train_oja_with_bias_unsupervised(teacher_network, X_train, gain_to_add)
elif "3_factor_hebbian" in experiment_name:
    weights, bias, eta = train_3factor_hebbian(teacher_network, X_train)

print("mean weight:", weights.mean(), weights.squeeze()[neurons_tuned_stim0].mean(), weights.squeeze()[neurons_tuned_stim1].mean())

weights, bias, flipped = orient_weights(weights, bias, X_train, y_train, crit=crit)

#test the learning rule (what's the accuracy?)
valid_acc, class_stim0, class_stim1 = test_learning_rule(X_test, y_test, weights, bias, crit=crit)

#test what happens upon lesioning
acc_lesioning = []; class_stim0_arr = []; class_stim1_arr = []
p_arr = [0.0001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 0.9]

for p in p_arr:
    val_acc, class_stim0, class_stim1 = find_bias_simple(p, weights, bias, X_test, y_test, crit=crit, seed=seed)
    acc_lesioning.append(torch.tensor(val_acc)); class_stim0_arr.append(torch.tensor(class_stim0)); class_stim1_arr.append(torch.tensor(class_stim1))

#print("------ generate *toy* new data to test heuristics principle --------")
#X_gen, y_gen = generate_data_main(X_test)
#rand_ind = np.random.permutation(X_gen.shape[1])
#X_gen = X_gen[:,rand_ind];
#test_learning_rule(X_test, y_test, weights, bias)

cp, CP_arr, neurons_tuned_list, prefs, choice = compute_cp(weights, bias, X_test, y_test, tunings, crit)
readout_weights, w_raw, neurons, explained, cp_rw, rho = compute_readout_weights(weights, bias, X_test, y_test, tunings, crit)

d_prime = compute_d_prime(X_test, y_test, eps=1e-8)
r_d_prime_cp, binned = compute_dprime_cp_correlation(d_prime, cp, neurons_tuned_list, prefs)
data_d = [binned[0]["d"], binned[1]["d"]]; data_c = [binned[0]["cp"], binned[1]["cp"]]

additional = "_no_centering" if centering == False else "_centering"
file_ending = "_crit_" + crit + "_p_over_" + str(p_over) + "_"+ parameter_list["W_init"] + "_noise_gp_co_ind_" + str(noise_group) + "_" + str(noise_common) + "_" + str(noise_independent) + "_seed_" + str(seed) + "_phase" + phase + additional + ".pt"

learning_rule_dir = learning_rule + "_new_model"
torch.save(torch.tensor(weights), "results/" + learning_rule_dir + "/weights" + file_ending)
torch.save(torch.tensor(bias), "results/" + learning_rule_dir + "/bias" + file_ending)
torch.save(valid_acc, "results/" + learning_rule_dir + "/val_acc" + file_ending)
torch.save(acc_lesioning, "results/" + learning_rule_dir + "/acc_lesioning" + file_ending)
torch.save(class_stim0_arr, "results/" + learning_rule_dir + "/class_stim0_arr" + file_ending)
torch.save(class_stim1_arr, "results/" + learning_rule_dir + "/class_stim1_arr" + file_ending)
torch.save(cp, "results/" + learning_rule_dir + "/cp" + file_ending)
torch.save(readout_weights, "results/" + learning_rule_dir + "/readout_weights" + file_ending)
torch.save(tunings[0], "results/" + learning_rule_dir + "/tuning_stim0" + file_ending)
torch.save(tunings[1], "results/" + learning_rule_dir + "/tuning_stim1" + file_ending)

#print("======= plot stuff ===========")
#plot_weights(teacher_network, weights, bias, X_train, tunings, cp, readout_weights, new_tunings, r_d_prime_cp, data_d, data_c, experiment_name + "_eta" + str(eta) + extra_name)