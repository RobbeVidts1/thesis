import numpy as np
import random
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time

import mags
from mags import calc_m


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
    if trans_rate_up + trans_rate_down > 0.9:
        print(trans_rate_up + trans_rate_down)

    if r < trans_rate_down:
        N_up -= 1
        return N_up
    elif r > 1 - trans_rate_up:
        N_up += 1
        return N_up
    else:
        return N_up

@jit()
def state_update_alt(N_up, N, beta, field, dt):
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

    trans_rate_up = dt * (N - N_up) * min(1, np.exp(beta * (original_energy - energy_up)))
    trans_rate_down = dt * N_up * min(1, np.exp(beta * (original_energy - energy_down)))
    r = random.random()
    if trans_rate_up + trans_rate_down > 0.9:
        print(trans_rate_up + trans_rate_down)

    if r < trans_rate_down:
        N_up -= 1
        return N_up
    elif r > 1 - trans_rate_up:
        N_up += 1
        return N_up
    else:
        return N_up


@jit()
def average_magnetization_heat_alt(run_number, N, h, beta, dt):
    """
    Finds the average magnetization and standard deviation using MCMC
    :param run_number: number of independent runs to average over
    :param N:
    :param h: array(len(time_arr)) external field
    :param beta: array(len(time_arr)) inverse temp
    :param dt: timestep. Required for computing the correct rate
    :return: array(4, len(time_arr)) containing m and the standard deviation on m, J and the standard deviation on J
    """

    m = np.zeros_like(h)
    mstdd = np.zeros_like(h)
    J = np.zeros_like(h)
    Jstdd = np.zeros_like(h)

    for run in range(run_number):
        N_up = random.randint(0, N)

        # for i=0, I don't calculate the heat, since I don't have a derivative
        N_up = state_update_alt(N_up, N, beta[0], h[0], dt)
        m[0] += (2 * N_up - N) / N
        mstdd[0] += ((2 * N_up - N) / N) ** 2

        for i in range(1, len(h)):
            N_up = state_update_alt(N_up, N, beta[i], h[i], dt)
            m[i] += (2 * N_up - N) / N / run_number
            mstdd[i] += ((2 * N_up - N) / N) ** 2 / run_number

            J[i] += (m[i] + m[i - 1] + h[i] + h[i - 1]) / 2 * (m[i] - m[i - 1]) / dt / run_number
            Jstdd[i] += ((m[i] + m[i - 1] + h[i] + h[i - 1]) / 2 * (m[i] - m[i - 1]) / dt) ** 2 / run_number

    mstdd = np.sqrt(mstdd - (m * m))

    Jstdd = np.sqrt(Jstdd - (J * J))
    return [m, mstdd, J, Jstdd]


@jit()
def average_magnetization_heat(run_number, N, h, beta, dt):
    """
    Finds the average magnetization and standard deviation using MCMC
    :param run_number: number of independent runs to average over
    :param N:
    :param h: array(len(time_arr)) external field
    :param beta: array(len(time_arr)) inverse temp
    :param dt: timestep. Required for computing the correct rate
    :return: array(4, len(time_arr)) containing m and the standard deviation on m, J and the standard deviation on J
    """

    m = np.zeros_like(h)
    mstdd = np.zeros_like(h)
    J = np.zeros_like(h)
    Jstdd = np.zeros_like(h)

    for run in range(run_number):
        N_up = 3*N//4

        # for i=0, I don't calculate the heat, since I don't have a derivative
        N_up = state_update(N_up, N, beta[0], h[0], dt)
        m[0] += (2 * N_up - N) / N
        mstdd[0] += ((2 * N_up - N) / N) ** 2

        for i in range(1, len(h)):
            N_up = state_update(N_up, N, beta[i], h[i], dt)
            m[i] += (2 * N_up - N) / N / run_number
            mstdd[i] += ((2 * N_up - N) / N) ** 2 / run_number

            J[i] += (m[i] + m[i - 1] + h[i] + h[i - 1]) / 2 * (m[i] - m[i - 1]) / dt / run_number
            Jstdd[i] += ((m[i] + m[i - 1] + h[i] + h[i - 1]) / 2 * (m[i] - m[i - 1]) / dt) ** 2 / run_number

    mstdd = np.sqrt(mstdd - (m * m))

    Jstdd = np.sqrt(Jstdd - (J * J))
    return [m, mstdd, J, Jstdd]


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

    beta_arr = beta_0 + epsilon * np.sin(omega_b*time_arr)
    m_0 = np.zeros_like(h)
    J_0 = np.zeros_like(h)
    m = np.zeros_like(h)
    J = np.zeros_like(h)
    C_avg = 0
    C_stdd = 0
    C_arr = np.empty(run_number)

    l = len(h)

    for run in range(run_number):
        N_up_0 = random.randint(0, N)
        N_up = N_up_0

        # for i=0, I don't calculate the heat, since I don't have a derivative
        N_up_0 = state_update(N_up_0, N, beta_0, h[0], dt)
        N_up = state_update(N_up, N, beta_arr[0], h[0], dt)
        m_0[0] = (2 * N_up_0 - N) / N
        m[0] = (2 * N_up - N) / N

        for i in range(1, l):
            N_up_0 = state_update(N_up_0, N, beta_0, h[i], dt)
            N_up = state_update(N_up, N, beta_arr[i], h[i], dt)

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
def Heat_capacity(beta_array, N, h_0, omega_0, ratio, epsilon, run_number, dt):
    omega_beta = omega_0 / ratio

    t_final = 2 * np.pi / omega_beta
    t_start = 2 * np.pi / omega_0

    time_arr = np.arange(-t_start, t_final, dt)
    l = len(time_arr)

    h = h_0 * np.sin(omega_0 * time_arr)

    beta_len = len(beta_array)

    C = np.empty((2, beta_len))
    for i_beta in range(beta_len):
        beta_0 = beta_array[i_beta]

        beta = beta_0 * (1 + epsilon * np.sin(omega_beta * time_arr))

        result_0 = average_magnetization_heat(run_number, N, h, beta_0 * np.ones_like(time_arr), dt)
        result_1 = average_magnetization_heat(run_number, N, h, beta, dt)

        J_1 = (result_1[2] - result_0[2]) / epsilon
        J_1stdd = np.sqrt(result_1[3] ** 2 + result_0[3] ** 2) / epsilon

        C[0][i_beta] = (beta_0 / np.pi * sum(
            J_1[l // (ratio + 1):] *
            np.cos(omega_beta * time_arr[l // (ratio + 1):]))
                        * dt)
        C[1][i_beta] = beta_0 / np.pi * np.sqrt(sum(
            J_1stdd[l // (ratio + 1):] ** 2 *
            np.cos(omega_beta * time_arr[l // (ratio + 1):]) ** 2
        )) * dt
        print('I have done '+str(i_beta)+'out of '+str(len(beta_array)))

    return C


def plotm_t_h(m, h, time_arr, N, omega_0, h_0, beta, final_time, T_amt, variance=False, run_number=0):
    """
    :param m:
    :param h:
    :param time_arr:
    :param N:
    :param omega_0:
    :param h_0:
    :param beta:
    :param final_time:
    :param T_amt:
    :param variance: if true, this only works when T_amt = 2
    :param run_number:
    :return:
    """
    l = len(time_arr)

    fig, (axleft, axright) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axleft.plot(time_arr[l // T_amt:], h[l // T_amt:], label='h', c='black')
    title = ('magnetization for ' + str(N) + ' spins, $h_0=$' + str(h_0) + r', $\beta = $' + str(beta) +
             r', $\omega_0 = $' + str(omega_0))
    axleft.set_xlabel('t')

    if variance == False:
        axleft.plot(time_arr[l // T_amt:], m[l // T_amt:], label='m', c='blue')

        points = np.array([h[l // T_amt:], m[l // T_amt:]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(time_arr[-1] / T_amt, time_arr[-1])
        lc = LineCollection(segs, cmap='jet', norm=norm, alpha=0.7, lw=0.95)
        lc.set_array(time_arr[l // T_amt:])

        axright.add_collection(lc)
    else:
        axleft.plot(time_arr[l // 2:], m[0][l // 2:], label='m')
        axleft.fill_between(time_arr[l // 2:], m[0][l // 2:] - (m[1][l // 2:] / 2),
                            m[0][l // 2:] + (m[1][l // 2:] / 2), alpha=0.5, facecolor='grey')

        axright.plot(h[l // 2:], m[0][l // 2:])
        axright.fill_between(h[l // 2:(l * 5) // 8],
                             m[0][l // 2:(l * 5) // 8] -
                             (m[1][l // 2:(l * 5) // 8] / 2),
                             m[0][l // 2:(l * 5) // 8] +
                             (m[1][l // 2:(l * 5) // 8] / 2), alpha=0.3, facecolor='grey')

        axright.fill_between(h[((l * 5) // 8) + 1:(l * 7) // 8],
                             m[0][((l * 5) // 8) + 1:(l * 7) // 8] -
                             (m[1][((l * 5) // 8) + 1:(l * 7) // 8] / 2),
                             m[0][((l * 5) // 8) + 1:(l * 7) // 8] +
                             (m[1][((l * 5) // 8) + 1:(l * 7) // 8] / 2), alpha=0.3, facecolor='grey')

        axright.fill_between(h[((l * 7) // 8) + 1:],
                             m[0][((l * 7) // 8) + 1:] -
                             (m[1][((l * 7) // 8) + 1:] / 2),
                             m[0][((l * 7) // 8) + 1:] +
                             (m[1][((l * 7) // 8) + 1:] / 2), alpha=0.3, facecolor='grey')

        title += r", $N_r = $" + str(run_number)

    axleft.legend(loc='lower right')
    fig.suptitle(title)

    axright.set_xlim(-h_0 * 1.05, h_0 * 1.05)
    axright.set_ylim(-1.05, 1.05)
    axright.set_xlabel('h')
    axright.set_ylabel('m')



def plotJ_t(results, time_arr, h, beta, N, omega_0, h_0, beta_0, ratio, run_number=1):
    l = len(time_arr)

    figJ, (axJup, axJdown) = plt.subplots(2, 1, layout='constrained', sharex=True)

    figJ.suptitle(
        'heat current for ' + str(N) + ' spins, $h_0=$' + str(h_0) + r', $\beta_0 = $' + str(beta_0) +
        r', $\omega_0 = $' + str(omega_0) + ', $N_r = $' + str(run_number))

    axJdown.plot(time_arr[l // (ratio + 1):], results[2][l // (ratio + 1):], alpha=0.5)
    axJdown.fill_between(time_arr[l // (ratio + 1):],
                         results[2][l // (ratio + 1):] - (results[3][l // (ratio + 1):] / 2),
                         results[2][l // (ratio + 1):] + (results[3][l // (ratio + 1):] / 2),
                         alpha=0.3,
                         color='grey'
                         )
    axJdown.set_title('heat current')
    axJdown.set_xlabel('t')
    axJdown.set_ylabel('J')

    axJup.plot(time_arr[l // (ratio + 1):], results[0][l // (ratio + 1):], label='$m$')
    axJup.fill_between(time_arr[l // (ratio+1):],
                       results[0][l // (ratio + 1):] - (results[1][l // (ratio + 1):]/2),
                       results[0][l // (ratio + 1):] + (results[1][l // (ratio + 1):]/2),
                       alpha=0.3, facecolor='grey')
    # axJup.plot(time_arr[l // (ratio + 1):], beta[l // (ratio + 1):] - beta_0, label=r'$\Delta \beta$')
    axJup.plot(time_arr[l // (ratio + 1):], h[l // (ratio + 1):], c='black', label=r'$h$')
    axJup.legend(loc='lower left')
    axJup.set_title('magnetization')


def plotJ_0_J_1(result_0, result_1, epsilon, time_arr, h, beta, N, omega_0, h_0, beta_0, ratio, run_number=1):
    l = len(time_arr)

    figJ1, (axJ1up, axJ1down) = plt.subplots(2, 1, layout='constrained', sharex=True)

    figJ1.suptitle(
        '$J_1$ for ' + str(N) + ' spins, $h_0=$' + str(h_0) + r', $\beta_0 = $' + str(beta_0) +
        r', $\omega_0 = $' + str(omega_0) + ', $N_r = $' + str(run_number))

    J_1 = (result_1[2] - result_0[2]) / epsilon
    J_1stdd = np.sqrt(result_1[3] ** 2 + result_0[3] ** 2) / epsilon

    axJ1down.plot(time_arr[l // (ratio + 1):], J_1[l // (ratio + 1):], alpha=0.6)
    axJ1down.fill_between(time_arr[l // (ratio + 1):],
                          J_1[l // (ratio + 1):] - J_1stdd[l // (ratio + 1):] / 2,
                          J_1[l // (ratio + 1):] + J_1stdd[l // (ratio + 1):] / 2,
                          alpha=0.3, color='grey')
    axJ1down.set_title('$J_1$')
    axJ1down.set_xlabel('$t$')
    axJ1down.set_ylabel('$J$')

    axJ1up.plot(time_arr[l // (ratio + 1):], result_1[0][l // (ratio + 1):], label=r'$m$')
    axJ1up.plot(time_arr[l // (ratio + 1):], h[l // (ratio + 1):], alpha=0.5,  label=r'$h$')
    axJ1up.plot(time_arr[l // (ratio + 1):], beta[l // (ratio + 1):] - beta_0, alpha=0.5, label=r'$\Delta \beta$')
    axJ1up.plot(time_arr[l // (ratio + 1):], result_0[0][l // (ratio + 1):], alpha=0.5, label=r'$m_{\epsilon=0}$')


    axJ1up.legend(loc='lower left')
    axJ1up.set_title('magnetization')


def compare_m():
    N = 50
    h_0 = 0.3
    beta_0 = 1.0
    omega_0 = 0.02
    ratio = 1

    omega_beta = omega_0 / ratio

    dt = 1 / (4 * N)

    run_number = 5000

    t_start = -2 * np.pi / omega_0
    t_final = 2 * np.pi / omega_beta
    time_arr = np.arange(t_start, t_final, dt)

    l = len(time_arr)

    h = h_0 * np.sin(omega_0 * time_arr)

    timer = time.time()
    m = average_magnetization_heat(run_number, N, h, beta_0 * np.ones_like(time_arr), dt)
    print(time.time() - timer)
    # m2 = average_magnetization_heat_alt(run_number, N, h, beta_0 * np.ones_like(time_arr), dt)

    fig, (axleft, axright) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axleft.plot(time_arr[l // 2:], h[l // 2:], label='h', c='black')
    title = ('Magnetization for ' + str(N) + ' spins, $h_0=$' + str(h_0) + r', $\beta = $' + str(beta_0) +
             r', $\omega_0 = $' + str(omega_0))
    axleft.set_xlabel('t')

    q = len(time_arr[l // 2:])
    axleft.plot(time_arr[l // 2:], m[0][l // 2:], label='m')
    axleft.fill_between(time_arr[l // 2:], m[0][l // 2:] - (m[1][l // 2:] / 2),
                        m[0][l // 2:] + (m[1][l // 2:] / 2), alpha=0.3, facecolor='grey')

    # # This is the alternate part ##############################################
    # axleft.plot(time_arr[l // 2:], m2[0][l // 2:], label='m_alt', c='red', alpha=0.7)
    # axleft.fill_between(time_arr[l // 2:], m2[0][l // 2:] - (m2[1][l // 2:] / 2),
    #                     m2[0][l // 2:] + (m2[1][l // 2:] / 2), alpha=0.4, facecolor='red')
    #
    # axright.plot(h[l // 2:], m2[0][l // 2:], color='red', alpha=0.4)
    # axright.fill_between(h[l // 2:(l * 5) // 8],
    #                      m2[0][l // 2:(l * 5) // 8] -
    #                      (m2[1][l // 2:(l * 5) // 8] / 2),
    #                      m2[0][l // 2:(l * 5) // 8] +
    #                      (m2[1][l // 2:(l * 5) // 8] / 2), alpha=0.2, facecolor='red')
    #
    # axright.fill_between(h[((l * 5) // 8) + 1:(l * 7) // 8],
    #                      m2[0][((l * 5) // 8) + 1:(l * 7) // 8] -
    #                      (m2[1][((l * 5) // 8) + 1:(l * 7) // 8] / 2),
    #                      m2[0][((l * 5) // 8) + 1:(l * 7) // 8] +
    #                      (m2[1][((l * 5) // 8) + 1:(l * 7) // 8] / 2), alpha=0.2, facecolor='red')
    #
    # axright.fill_between(h[((l * 7) // 8) + 1:],
    #                      m2[0][((l * 7) // 8) + 1:] -
    #                      (m2[1][((l * 7) // 8) + 1:] / 2),
    #                      m2[0][((l * 7) // 8) + 1:] +
    #                      (m2[1][((l * 7) // 8) + 1:] / 2), alpha=0.2, facecolor='red')
    # ###########

    axright.plot(h[l // 2:], m[0][l // 2:])
    axright.fill_between(h[l // 2:(l * 5) // 8],
                         m[0][l // 2:(l * 5) // 8] -
                         (m[1][l // 2:(l * 5) // 8] / 2),
                         m[0][l // 2:(l * 5) // 8] +
                         (m[1][l // 2:(l * 5) // 8] / 2), alpha=0.3, facecolor='grey')

    axright.fill_between(h[((l * 5) // 8) + 1:(l * 7) // 8],
                         m[0][((l * 5) // 8) + 1:(l * 7) // 8] -
                         (m[1][((l * 5) // 8) + 1:(l * 7) // 8] / 2),
                         m[0][((l * 5) // 8) + 1:(l * 7) // 8] +
                         (m[1][((l * 5) // 8) + 1:(l * 7) // 8] / 2), alpha=0.3, facecolor='grey')

    axright.fill_between(h[((l * 7) // 8) + 1:],
                         m[0][((l * 7) // 8) + 1:] -
                         (m[1][((l * 7) // 8) + 1:] / 2),
                         m[0][((l * 7) // 8) + 1:] +
                         (m[1][((l * 7) // 8) + 1:] / 2), alpha=0.3, facecolor='grey')

    title += r", $N_r = $" + str(run_number)

    fig.suptitle(title)

    axright.set_xlim(-h_0 * 1.05, h_0 * 1.05)
    axright.set_ylim(-1.05, 1.05)
    axright.set_xlabel('h')
    axright.set_ylabel('m')

    dt = 1e-4

    smooth_m = mags.calc_m(h_0, omega_0, beta_0, dt, t_final-t_start, omega_beta=omega_beta)

    time_arr = np.linspace(0, t_final-t_start, int((t_final-t_start)/dt))
    h = h_0 * np.sin(omega_0 * time_arr)

    l=len(time_arr)

    axleft.plot(smooth_m[0][l//2:]+t_start, smooth_m[1][l//2:], c='red', label='macroscopic', alpha=1)
    axright.plot(h[l//2:], smooth_m[1][l//2:], c='red', alpha=0.7)

    axleft.legend(loc='lower right')


def compare_N():
    N_arr = np.array([3, 10, 50, 200])
    l_n = len(N_arr)
    h_0 = 0.3
    beta_0 = 2.5
    omega_0 = 0.02
    ratio = 1
    Relax_time = 2
    omega_beta = omega_0/ratio
    run_number = 10000

    t_start = -2 * np.pi / omega_0 * Relax_time
    t_final = 2 * np.pi / omega_beta

    fig, (axleft, axright) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout='constrained')


    for n_i in range(l_n):
        N = N_arr[n_i]

        dt = dt = 1 / (2 * N)
        time_arr = np.arange(t_start, t_final, dt)
        l = len(time_arr)
        h = h_0 * np.sin(omega_0 * time_arr)
        result = average_magnetization_heat(run_number, N, h, beta_0 * np.ones_like(time_arr), dt)

        axleft.plot(time_arr[int(l*(Relax_time / (ratio+Relax_time))):], result[0][int(l*(Relax_time / (ratio+Relax_time))):], label=('$N = $'+str(N)))

        axright.plot(h[int(l*(Relax_time / (ratio+Relax_time))):], result[0][int(l*(Relax_time / (ratio+Relax_time))):])



    axleft.plot(time_arr[int(l*(Relax_time / (ratio+Relax_time))):], h[int(l*(Relax_time / (ratio+Relax_time))):], label='h', c='black', alpha=0.7)

    fig.suptitle(r'Effect of Spin Number on Magnetization at $\beta=2.5$, $h_0=0.3$ and $\omega_0=0.02$')
    axleft.legend(loc='lower left')
    axleft.set_xlabel('$t$')
    axleft.set_ylabel('$m$')
    axright.set_xlabel('$h$')



def main():
    # N = 50
    # h_0 = 0.3
    # # beta_0 = 1.8
    # omega_0 = 0.02
    # ratio = 50
    # epsilon = 0.1
    #
    # omega_beta = omega_0 / ratio
    #
    # dt = 1 / (2 * N)
    #
    # run_number = 250
    #
    # t_start = -2 * np.pi / omega_0
    # t_final = 2 * np.pi / omega_beta
    # time_arr = np.arange(t_start, t_final, dt)
    #
    # # l = len(time_arr)
    #
    # h = h_0 * np.sin(omega_0 * time_arr)
    # # beta = beta_0 + epsilon * np.sin(omega_beta * time_arr)
    #
    # # timer = time.time()
    # # result_0 = average_magnetization_heat(run_number, N, h, beta_0 * np.ones_like(time_arr), dt)
    # # result_1 = average_magnetization_heat(run_number, N, h, beta, dt)
    # # print(time.time() - timer)
    #
    # # plotm_t_h(result_0[:2], h, time_arr, N, omega_0, h_0, beta_0, 10, ratio + 1, variance=True, run_number=run_number)
    #
    # # plotJ_t(result_0, time_arr, h, beta, N, omega_0, h_0, beta_0, ratio, run_number=run_number)
    #
    # # plotJ_0_J_1(result_0, result_1, epsilon, time_arr, h, beta, N, omega_0, h_0, beta_0, ratio, run_number=run_number)
    #
    # beta_array = np.linspace(0.05, 3.5, 40)
    # C = np.empty((len(beta_array),2))
    # # timer = time.time()
    # # C = Heat_capacity(beta_array, N, h_0, omega_0, ratio, epsilon, run_number, dt)
    # # print(time.time() - timer)
    # for i in range(len(beta_array)):
    #     C[i] = average_heat_cap(run_number, N, h, beta_array[i], omega_beta, epsilon, time_arr, ratio, dt)
    #     print(i)
    #
    # fig, ax = plt.subplots(layout='tight')
    #
    # ax.plot(beta_array, C[:,0])
    # ax.fill_between(beta_array, C[:,0] - C[:,1] / 2, C[:,0] + C[:,1] / 2, alpha=0.3, facecolor='grey')
    # ax.set_title('heat capacity for ' + str(N) + ' spins, $h_0=$' + str(h_0) +
    #           r', $\omega_0 = $' + str(omega_0) + ', $N_r = $' + str(run_number))
    # ax.set_ylabel('$C/k_B$')
    # ax.set_xlabel(r'$\beta$')
    # ax.axhline()
    #
    # ax.set_ylim(-1.5,1)
    #
    # # compare_m()
    compare_N()
    plt.show()


if __name__ == '__main__':
    main()
