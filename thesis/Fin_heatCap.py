#!/usr/bin/env ['python3']

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-user=robbe.vidts@student.kuleuven.be
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

import numpy as np
import random
from numba import jit
import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
import time



# I will assume J = nu = 1
@jit()
def energy(N_up, N, field):
    """
    gives back the energy of the state
    :param N_up: int between 0 and N, denoting the number of spins pointing up.
    :param field: float applying the field
    :param N: int: number of spins
    :return: float: the energy
    """
    m = (2 * N_up - N) / N
    return -N * (m ** 2 / 2 + field * m)


@jit()
def state_update(N_up, N, beta, field, dt):
    """
    :param N_up:
    :param N:
    :param beta: float
    :param field: float
    :param dt: Should always be smaller than 1/N
    :return:
    """

    original_energy = energy(N_up, N, field)
    energy_up = energy(N_up + 1, N, field)
    energy_down = energy(N_up - 1, N, field)

    # trans_rate_up = dt * (N - N_up) * min(1, np.exp(beta * (original_energy - energy_up)))
    # trans_rate_down = dt * N_up * min(1, np.exp(beta * (original_energy - energy_down)))
    trans_rate_up = dt * (N - N_up) / (1 + np.exp(beta * (-original_energy + energy_up)))
    trans_rate_down = dt * N_up / (1 + np.exp(beta * (-original_energy + energy_down)))
    r = random.random()
    # if trans_rate_up + trans_rate_down > 0.1:
    #     # print(trans_rate_up + trans_rate_down)
    #     # print('Oh boy, a little too fast')
    if r < trans_rate_down:
        N_up -= 1
        return N_up
    elif r > 1 - trans_rate_up:
        N_up += 1
        return N_up
    else:
        return N_up


@jit()
def average_heat_cap(run_number, N, h, beta_0, omega_b, epsilon, time_arr, ratio, dt):
    """
    Finds the average magnetization and standard deviation using MCMC
    :param run_number: number of independent runs to average over
    :param N:
    :param h: array(len(time_arr)) external field
    :param beta: array(len(time_arr)) inverse temp
    :param dt: timestep. Required for computing the correct rate
    :return: array(4, len(time_arr)) containing m and the standard deviation on m, J and the standard deviation on J
    """

    beta_t = beta_0*(1 + epsilon * np.sin(omega_b*time_arr))
    m_0 = np.zeros_like(h)
    J_0 = np.zeros_like(h)
    m = np.zeros_like(h)
    J = np.zeros_like(h)
    # C_avg = 0
    # C_stdd = 0
    C_arr = np.empty(run_number)

    l = len(h)

    for run in range(run_number):
        N_up_0 = random.randint(0, N)
        N_up = N_up_0

        # for i=0, I don't calculate the heat, since I don't have a derivative
        N_up_0 = state_update(N_up_0, N, beta_0, h[0], dt)
        N_up = state_update(N_up, N, beta_t[0], h[0], dt)
        m_0[0] = (2 * N_up_0 - N) / N
        m[0] = (2 * N_up - N) / N

        for i in range(1, l):
            N_up_0 = state_update(N_up_0, N, beta_0, h[i], dt)
            N_up = state_update(N_up, N, beta_t[i], h[i], dt)

            m_0[i] = (2 * N_up_0 - N) / N
            m[i] = (2 * N_up - N) / N

            J_0[i] = (m_0[i] + m_0[i - 1] + h[i] + h[i - 1]) / 2 * (m_0[i] - m_0[i - 1]) / dt
            J[i] = (m[i] + m[i - 1] + h[i] + h[i - 1]) / 2 * (m[i] - m[i - 1]) / dt

        C_arr[run] = (beta_0 / np.pi / epsilon * sum(
            (J[l // (ratio + 1):] - J_0[l // (ratio + 1):]) *
            np.cos(omega_b * time_arr[l // (ratio + 1):]))
                        * dt)

    C_avg = np.average(C_arr)
    C_stdd = np.std(C_arr)
    return [C_avg, C_stdd]

@jit()
def C_beta(beta_array, run_number, N, h, omega_beta, epsilon, time_arr, ratio, dt):
    res = np.empty((len(beta_array),2))
    # timer = time.time()
    for i in range(len(beta_array)):
        res[i] = average_heat_cap(run_number, N, h, beta_array[i], omega_beta, epsilon, time_arr, ratio, dt)
        print(i)
        # if i == 0:
            # print(time.time()-timer)
    return res

def main():
    N = 50
    h_0 = 0.3
    omega_0 = 0.02
    ratio = 75
    epsilon = 0.2

    omega_beta = omega_0 / ratio

    dt = 1 / (4 * N)

    run_number = 5000

    t_start = -2 * np.pi / omega_0
    t_final = 2 * np.pi / omega_beta
    time_arr = np.arange(t_start, t_final, dt)

    h = h_0 * np.sin(omega_0 * time_arr)

    beta_array = np.linspace(0.1, 4.0, 50)

    C = C_beta(beta_array, run_number, N, h, omega_beta, epsilon, time_arr, ratio, dt)

    np.savetxt("C_50a.csv", C, delimiter=",")

    fig, ax = plt.subplots(layout='tight')

    ax.plot(beta_array, C[:,0])
    ax.fill_between(beta_array, C[:,0] - C[:,1] / 2, C[:,0] + C[:,1] / 2, alpha=0.3, facecolor='grey')
    ax.set_title('Heat Capacity for ' + str(N) + ' spins, $h_0=$' + str(h_0) +
              r', $\omega_0 = $' + str(omega_0) + ', $N_r = $' + str(run_number) + ', $r = $' + str(ratio)
                 + r', $\epsilon=$'+str(epsilon), wrap=True)
    ax.set_ylabel('$C/k_B$')
    ax.set_xlabel(r'$J\beta$')
    ax.grid(visible=True, axis='both', linewidth=0.5, alpha=0.3, color='k')
    # ax.axhline(c='black', linestyle='dashed', alpha=0.5)

    ax.set_ylim(-2.5,2.5)

    plt.show()



if __name__ == '__main__':
    main()
