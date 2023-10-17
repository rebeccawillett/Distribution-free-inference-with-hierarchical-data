import torch
import numpy as np
import pdb, random
import argparse, os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz96(t, x, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i] + F
    return dxdt

def generate_l96_data(hyperparams, N=40):
    ## hyperparams: [seed, F, T, dt]
    seed = int(hyperparams[0])
    np.random.seed(seed)
    F = hyperparams[1]
    T = hyperparams[2]
    dt = hyperparams[3]
    N = hyperparams[4]
    t_span = (0, T)
    initial_conditions = hyperparams[-int(N):]
    if initial_conditions.sum() == 0:
        initial_conditions = np.random.normal(0, 1, int(N))
    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(
        lorenz96,
        t_span,
        initial_conditions,
        args=(F,),
        t_eval=t_eval,
        method='DOP853',
        rtol=1e-6,  # Relative tolerance
        atol=1e-9,  # Absolute tolerance
    )
    return sol.y.T

def create_data_folder(data_folder):
    ## for creating the data folder
    os.mkdir(data_folder) if not os.path.exists(data_folder) else print('exist')

def save_l96_data(data_folder, dt, T_1, T_2, num_of_samples_k, num_of_perturbations_n, eta_perturb, F_param = 10, N = 40):

    seeds_base = data_folder[1]
    data_folder_name = data_folder[0]
    create_data_folder('data')
    seeds = seeds_base + np.arange(num_of_samples_k).reshape(num_of_samples_k, 1)
    T_array = np.repeat(np.array([T_1])[None, :], num_of_samples_k, axis = 0)
    params_array = np.repeat(np.array([F_param])[None, :], num_of_samples_k, axis = 0)
    dt_array = np.repeat(np.array([dt])[None, :], num_of_samples_k, axis = 0)
    ## when all the components are zero, we randomly sample initial initial_conditions from normal distributions
    initial_conditions_array = np.repeat(np.zeros(N)[None, :], num_of_samples_k, axis = 0)
    N_array = np.repeat(np.array([N])[None, :], num_of_samples_k, axis = 0)
    hyperparams = np.concatenate([seeds, params_array, T_array, dt_array, N_array, initial_conditions_array], axis = 1)
    n_workers = 50
    with Pool(n_workers) as pool:
        total_traj = np.array(pool.map(generate_l96_data, hyperparams))
    ## set the initial conditions for each perturbed sample in the group
    initial_conditions_array = np.repeat(total_traj[:, -1, :][:, None, :], num_of_perturbations_n, axis = 1)
    ## sample the noise
    initial_conditions_array_noise = eta_perturb * np.random.normal(0, 1, num_of_samples_k * num_of_perturbations_n * N).reshape(num_of_samples_k, num_of_perturbations_n, -1)
    initial_conditions_array_unperturbed = initial_conditions_array.copy()
    initial_conditions_array = initial_conditions_array + initial_conditions_array_noise
    initial_conditions_array = initial_conditions_array.reshape(num_of_samples_k * num_of_perturbations_n, -1)
    seeds = seeds_base + np.arange(num_of_samples_k * num_of_perturbations_n).reshape(num_of_samples_k * num_of_perturbations_n, 1)
    T_array = np.repeat(np.array([T_2])[None, :], num_of_samples_k * num_of_perturbations_n, axis = 0)
    params_array = np.repeat(np.array([F_param])[None, :], num_of_samples_k * num_of_perturbations_n, axis = 0)
    dt_array = np.repeat(np.array([dt])[None, :], num_of_samples_k * num_of_perturbations_n, axis = 0)
    N_array = np.repeat(np.array([N])[None, :], num_of_samples_k * num_of_perturbations_n, axis = 0)
    hyperparams = np.concatenate([seeds, params_array, T_array, dt_array, N_array, initial_conditions_array], axis = 1)
    with Pool(n_workers) as pool:
        total_traj = np.array(pool.map(generate_l96_data, hyperparams))
    total_traj = total_traj.reshape(num_of_samples_k, num_of_perturbations_n, int(T_2/dt), N)
    torch.save((total_traj, initial_conditions_array_unperturbed), f'data/{data_folder_name}.pth')
    print(f'saved {data_folder[0]}')
    return total_traj, initial_conditions_array_unperturbed

#train_data = save_l96_data(data_folder = ['train', 2000], dt = 0.1, T_1 = 20, T_2 = 10, num_of_samples_k = 10, num_of_perturbations_n = 20, eta_perturb = 0.01, N = 10)
#print(train_data[0].shape, train_data[1].shape)
