import numpy as np
import random
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time


# I will assume J = nu = 1
@jit()
def energy(N_up, field, N):
    """
    gives back the energy of the state
    :param m: int current number of up_spins
    :param field: float applying the field
    :param N: int: number of spins
    :return: float: the energy
    """
    m = (N_up - (N-N_up))/N
    return -N * (m ** 2 / 2 + field * m)


@jit()
def rates(N_up, beta, field, N, dt):
    """
    Gives the transition flip rates as in eq. V.I in the paper
    :param N_up: int number of up_spins
    :param beta:
    :param field:
    :return: array containing two floats, x[0] is rate for N_up +1, and x[1] rate for N_up-1
    """
    result = np.zeros(2)
    original_energy = energy(N_up, field, N)

    if N_up != N:
        energy_up = energy(N_up+1, field, N)
        result[0] = dt*N*(N-N_up)/(1+np.exp(beta * (original_energy - energy_up)))
    if N_up != 0:
        energy_down = energy(N_up-1, field, N)
        result[1] = dt*N*N_up/(1+np.exp(beta * (original_energy - energy_down)))

    return result


@jit()
def state_update(N_up, beta, field, N, dt):
    alpha_arr = rates(N_up, beta, field, N)
    r = random.random()

    # updating the number of up-spins. If r is large, then more up_spins, if r is small, les up-spins
    #  else nothing
    if r > 1-alpha_arr[0]:
        return N_up + 1
    elif r < alpha_arr[1]:
        return N_up - 1
    else:
        return N_up


@jit()
def solver(N, h_0, omega_0, beta_0, omega_beta, epsilon, init_time, exp_time):
    """
    Solves the finite spin model
    :param N: spin number
    :param h_0:
    :param omega_0:
    :param beta_0:
    :param omega_beta:
    :param epsilon:
    :param init_time: float: periods of omega_0 to use for initializing
    :param exp_time: float: periods of omega_beta to use for experiment # notice difference
    :return: array containing time, magnetization and energy
    """
    # I have some doubts about np.arange for such long lists of arrays
    # (potential big number + small number error or
    #                                      https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    dt = 5e-5/omega_0 # This is still open to change
    time_arr = np.arange(start=-(2*np.pi*init_time/omega_0), stop=2*np.pi*exp_time/omega_beta, step=dt)
    beta_arr = beta_0*(1+epsilon*np.sin(omega_beta*time_arr))
    field_arr = h_0*np.sin(omega_0*time_arr)

    # finding all the steps
    total_steps = len(time_arr)
    init_steps = int(total_steps/(exp_time*(omega_0/omega_beta))) - 10 ## making a guess to make the while loop short
                    # (only works well when init_time==1)
    while time_arr[init_steps + 1] < 0:
        init_steps += 1
    exp_steps = total_steps - init_steps

    #initialize open to change
    N_up = random.randint(0, N)

    result = np.empty((3, exp_steps))
    result[0] = time_arr[init_steps:]

    print(result[0][0])

    for i in range(init_steps):
        N_up = state_update(N_up, beta_arr[i], field_arr[i], N, dt)

    print(i)

    for i in range(exp_steps):
        N_up = state_update(N_up, beta_arr[init_steps + i], field_arr[init_steps + i], N, dt)
        result[1][i] = (2*N_up-N)/N
        result[2][i] = energy(N_up, field_arr[init_steps + i], N)

    return result


