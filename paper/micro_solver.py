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
        result[0] = dt*(N-N_up)/(1+np.exp(beta * (original_energy - energy_up)))
    if N_up != 0:
        energy_down = energy(N_up-1, field, N)
        result[1] = dt*N_up/(1+np.exp(beta * (original_energy - energy_down)))

    return result


@jit()
def state_update(N_up, beta, field, N, dt):
    alpha_arr = rates(N_up, beta, field, N, dt)
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
def solver(N, field_arr, beta_arr, time_arr, init_steps, dt):
    """
    Solves the finite spin model
    :param N: spin number
    :param field_arr: array containing the field strengths
    :param beta_arr: array containing time-dependent temperatures
    :param time_arr: array containg the time
    :return: array containing time, magnetization and energy, only of the experiment part (no equilibration)
    """
    # I have some doubts about np.arange for such long lists of arrays
    # (potential big number + small number error )
    #                                      https://numpy.org/doc/stable/reference/generated/numpy.arange.html


    #initialize open to change
    N_up = random.randint(0, N)

    exp_steps = len(time_arr - init_steps)
    result = np.empty((3, exp_steps))
    result[0] = time_arr[init_steps:]

    print(len(time_arr[init_steps:]))
    print(len(result[1]))

    print("This should be close to 0 " + str(result[0][0]))

    for i in range(init_steps):
        N_up = state_update(N_up, beta_arr[i], field_arr[i], N, dt)

    for i in range(exp_steps):
        N_up = state_update(N_up, beta_arr[init_steps + i], field_arr[init_steps + i], N, dt)
        result[1][i] = (2*N_up-N)/N
        result[2][i] = energy(N_up, field_arr[init_steps + i], N)

    return result


def Trial():
    x = solver(40, 0.7, 0.02, 3.0, 0.02, 0, 1.0, 1.0)

    fig1, ax1 = plt.subplots()
    ax1.plot(x[0], x[1])
    fig2, ax2 = plt.subplots()
    ax2.plot(x[0],x[2])
    plt.show()


def average_magnetization():
    # physical parameters
    N_1 = 50
    N_2 = 100

    h_0 = 0.3
    omega_0 = 0.02
    beta_0 = 1.3
    # compute parameters
    exp_time = 1
    init_time = 1

    dt_1 = 1/(2*N_1)  ## Perhaps we should check if this is indeed sufficient.
    dt_2 = 1/(2*N_2)

    run_number = 100

    # I have some doubts about np.arange for such long lists of arrays
    # (potential big number + small number error )
    #                                      https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    # creating the arrays for the first one
    time_arr = np.arange(start=-(2 * np.pi * init_time / omega_0), stop=2 * np.pi * exp_time / omega_0, step=dt_1)
    beta_arr = beta_0 * np.ones_like(time_arr)
    field_arr = h_0 * np.sin(omega_0 * time_arr)

    # finding all the step numbers
    total_steps = len(time_arr)
    init_steps = total_steps // 2 - 10 ## making a guess to make the while loop short
    while time_arr[init_steps + 1] < 0:
        init_steps += 1
    exp_steps = total_steps - init_steps

    result_matrix_1 = solver(N_1, field_arr, beta_arr, time_arr, init_steps, dt_1)










def main():
    average_magnetization()


if __name__ == '__main__':
    main()


