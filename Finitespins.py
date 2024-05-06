import numpy as np
import random
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time


# I will assume J = nu = 1
@jit()
def energy(state, field, N):
    """
    gives back the energy of the state
    :param state: array with +1/-1
    :param field: float applying the field
    :param N: int: number of spins
    :return: float: the energy
    """
    m = sum(state) / N
    return -N * (m ** 2 / 2 + field * m)


@jit()
def rate_simple(state, position, beta, field, N):
    """
    Gives the simple transition flip rate
    :param state:
    :param position:
    :param beta:
    :param field:
    :return:
    """

    original_energy = energy(state, field, N)
    state[position] *= -1
    new_energy = energy(state, field, N)
    state[position] *= -1

    return min(1, np.exp(beta * (original_energy - new_energy)))


@jit()
def state_update(state, beta, field, N, dt):
    position = random.randint(0, N - 1)

    # if random.random() < noise_level:
    #     state[position] = random.randrange(2) * 2 - 1
    #     return state

    alpha = dt * N * rate_simple(state, position, beta, field, N)

    if random.random() < alpha:
        state[position] *= -1
        return state
    else:
        return state


@jit()
def initialize(state, n):
    for i in range(n):
        if random.random() > 0.5:
            state[i] = -1
    return state


@jit()
def Heat_using_energy(energy, m, field, dt):
    """
    Calculates J = -dE/dt - m dh/dt
    :param energy:
    :param m:
    :param field:
    :param dt:
    :return:
    """
    result = np.empty(len(energy))

    for i in range(len(result)):
        if i == 0:
            result[i] = 0
        else:
            fieldchange = (field[i]-field[i-1])/dt
            result[i] = -(energy[i] - energy[i-1])/dt - fieldchange*(m[i]+m[i-1])/2
    return result

@jit()
def Heat_using_m(m, field, dt):
    """
    :param m: Assumed to also contain the variance
    :param field:
    :param dt:
    :return:
    """
    result = np.empty_like(m[0])
    for i in range(len(result)):
        if i==0:
            result[i] = 0
        else:
            result[i] = ( (m[0][i]+m[0][i-1])/2 + (field[i]+field[i-1])/2 ) * (m[0][i]-m[0][i-1]) /dt
    return result
    # result = np.empty_like(m)
    # for i in range(len(m[0])):
    #     if i==0:
    #         result[0][i] = 0
    #         result[1][i] = 0
    #     else:
    #         result[0][i] = ( (m[0][i]+m[0][i-1])/2 + (field[i]+field[i-1])/2 ) * (m[0][i]-m[0][i-1]) /dt
    #         result[1][i] = np.sqrt( (m[0][i]/dt * m[1][i])**2 + (m[0][i-1]/dt * m[1][i-1])**2 )
    # return result




def plotm_t_h(m, h, time_arr, spin_number, omega_0, h_0, beta, final_time, T_amt, variance=False, run_number=0):
    """
    
    :param m: 
    :param h: 
    :param time_arr: 
    :param spin_number: 
    :param omega_0: 
    :param h_0: 
    :param beta: 
    :param final_time: 
    :param T_amt: 
    :param variance: if true, this only works when T_amt = 2
    :param run_number: 
    :return: 
    """
    l=len(time_arr)

    fig, (axleft, axright) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axleft.plot(time_arr[l // T_amt:], h[l // T_amt:], label='h', c='orange')
    title = ('magnetization for ' + str(spin_number) + ' spins, $h_0=$' + str(h_0) + r', $\beta = $' + str(beta) +
             r', $\omega_0 = $' + str(omega_0))
    axleft.set_xlabel('t')

    if variance == False:
        axleft.plot(time_arr[l // T_amt:], m[l // T_amt:], label='m', c='blue')

        points = np.array([h[l // T_amt:], m[l // T_amt:]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(time_arr[-1]/T_amt, time_arr[-1])
        lc = LineCollection(segs, cmap='jet', norm=norm, alpha=0.7, lw=0.95)
        lc.set_array(time_arr[l // T_amt:])

        axright.add_collection(lc)
    else:
        q = len(time_arr[l // 2:])
        axleft.plot(time_arr[l // 2:], m[0][l // 2:], label='m', c='blue')
        axleft.fill_between(time_arr[l // 2:], m[0][l // 2:] - (m[1][l // 2:] /2),
                             m[0][l // 2:] + (m[1][l // 2:] /2), alpha=0.5)

        axright.plot(h[l // 2:], m[0][l // 2:])
        axright.fill_between(h[l // 2:(l*5)//8],
                             m[0][l // 2:(l*5)//8] -
                             (m[1][l // 2:(l*5)//8] /2),
                             m[0][l // 2:(l*5)//8] +
                             (m[1][l // 2:(l*5)//8] /2), alpha=0.25, facecolor='grey')

        axright.fill_between(h[((l * 5) // 8) +1 :(l * 7) // 8],
                             m[0][((l * 5) // 8) +1:(l * 7) // 8] -
                             (m[1][((l * 5) // 8) +1:(l * 7) // 8] /2),
                             m[0][((l * 5) // 8) +1:(l * 7) // 8] +
                             (m[1][((l * 5) // 8) +1:(l * 7) // 8] /2), alpha=0.25, facecolor='grey')

        axright.fill_between(h[((l * 7) // 8) + 1:],
                             m[0][((l * 7) // 8) + 1:] -
                             (m[1][((l * 7) // 8) + 1:] /2),
                             m[0][((l * 7) // 8) + 1:] +
                             (m[1][((l * 7) // 8) + 1:] /2), alpha=0.25, facecolor='grey')

        title += r", $N_r = $" + str(run_number)

    axleft.legend(loc='lower right')
    fig.suptitle(title)

    axright.set_xlim(-h_0*1.05, h_0*1.05)
    axright.set_ylim(-1.05, 1.05)
    axright.set_xlabel('h')
    axright.set_ylabel('m')


def Heat_and_plot_stat(spin_number, h, noise, time_arr, beta_0, omega_beta, epsilon, dt, T_amt):
    state = np.ones(spin_number)
    Energy = np.empty_like(time_arr)
    m = np.empty_like(time_arr)
    l = len(time_arr)

    for b in range(len(beta_array)):
        initialize(state, spin_number)
        for i in range(len(time_arr)):
            Energy[i] = energy(state, h[i], spin_number)
            state_update(state, beta_array[b], h[i], spin_number, noise, dt)
            m[i] = sum(state)

        Heat_current = Heat(Energy, m, h, dt)
        plotlabel = r'$\beta = $' + str(beta_array[b])
        ax3.plot(time_arr[l // T_amt:], Heat_current[l // T_amt:], label=plotlabel)

    ax3.legend(loc='upper right')

@jit()
def average_magnetization(run_number, spin_number, h, beta, dt):
    """
    Finds the average magnetization and standard deviation using MCMC
    :param run_number: number of independent runs to average over
    :param spin_number:
    :param h: array(len(time_arr)) external field
    :param beta: array(len(time_arr)) inverse temp
    :param noise: Some noise level if wanted
    :param time_arr: the time
    :param dt: timestep. Required for computing the correct rate
    :return: array(2, len(time_arr)) containing m and the standard deviation on m
    """
    state = np.ones(spin_number)
    m = np.zeros(len(h))
    mstdd = np.zeros(len(h))

    for run in range(run_number):
        initialize(state, spin_number)
        for i in range(len(h)):
            state_update(state, beta[i], h[i], spin_number, dt)
            m[i] += sum(state)/spin_number /run_number
            mstdd[i] += (sum(state)/spin_number)**2 /run_number
    mstdd = np.sqrt(mstdd-(m*m))

    return [m, mstdd]


@jit()
def Heat_split_up(spin_number, h, noise, time_arr, beta_0, omega_beta, epsilon, dt):
    """

    :param spin_number:
    :param h:
    :param noise:
    :param time_arr:
    :param beta_0:
    :param omega_beta:
    :param dt:
    :return: [j_0, J] both an array of length len(time_arr)
    """
    l=len(time_arr)
    state = np.ones(spin_number)
    Energy = np.empty_like(time_arr)
    m = np.empty_like(time_arr)

    # J_0
    initialize(state, spin_number)
    for i in range(len(time_arr)):
        Energy[i] = energy(state, h[i], spin_number)
        m[i] = sum(state)
        state_update(state, beta_0, h[i], spin_number, noise)

    J_0 = Heat(Energy, m, h, dt)

    #J
    initialize(state, spin_number)
    beta_t = beta_0 + epsilon * np.sin(omega_beta*time_arr)

    for i in range(len(time_arr)):
        Energy[i] = energy(state, h[i], spin_number)
        m[i] = sum(state)
        state_update(state, beta_t[i], h[i], spin_number, noise)

    J = Heat(Energy, m, h, dt)

    return [J_0, J]

@jit()
def Heat_capacity(beta_array, spin_number, h_0, omega_0, ratio, epsilon, run_number, dt):

    omega_beta = omega_0/ratio

    t_final = 2 * np.pi / omega_beta
    t_start = 2 * np.pi / omega_0

    time_arr = np.arange(-t_start, t_final, dt)
    l = len(time_arr)

    h = h_0 * np.sin(omega_0 * time_arr)


    C = np.empty_like(beta_array)
    for i_beta in range(len(beta_array)):
        beta_0 = beta_array[i_beta]

        beta_cst = beta_0 * np.ones_like(time_arr)
        m_0 = average_magnetization(run_number, spin_number, h, beta_cst, dt)

        beta = beta_0 * (1 + epsilon * np.sin(omega_beta * time_arr))
        m_eps = average_magnetization(run_number, spin_number, h, beta, dt)

        heat_1 = (Heat_using_m(m_eps, h, dt) - Heat_using_m(m_0, h, dt)) / epsilon

        C[i_beta] = beta_0/np.pi * sum(heat_1[l//(ratio+1):] * np.cos(omega_beta*time_arr[l//(ratio+1):])) * dt

    return C


def main():
    # parameters
    spin_number = 20
    omega_0 = 1e-1
    beta_array = np.linspace(0.1,3.5,50)
    h_0 = 0.3

    ratio=5 # The ratio omega_0/omega_beta

    epsilon = 1e-1
    omega_beta = omega_0/ratio

    run_number = 1e3

    dt = 1e-2

    start_time = time.time()

    C = Heat_capacity(beta_array, spin_number, h_0, omega_0, ratio, epsilon, run_number, dt)

    print(time.time()-start_time)

    plt.plot(beta_array, C)
    plt.title('Heat capacity for ' + str(spin_number) + r' spins, $h_0 = $' + str(h_0)  + r', $ \omega_0 = $'
              + str(omega_0) + r", $r = $" + str(ratio))
    plt.ylabel(r'C')
    plt.xlabel(r'$\beta$')

    # figx, axx = plt.subplots()
    #
    # axx.plot(time_arr[l//(ratio+1):], heat_1[l//(ratio+1):])
    # axx.plot(time_arr[l//(ratio+1):], beta[l//(ratio+1):] - beta_0)
    # # plt.fill_between(time_arr[l//T_amt:],
    # #                  heat[0][l//T_amt:] - heat[1][l//T_amt:]/2,
    # #                  heat[0][l//T_amt:] + heat[1][l//T_amt:]/2, alpha=0.25, facecolor='grey')
    # # title = (r'Heat current for ' + str(spin_number) + r' spins, $h_0 = $' + str(h_0) + r', $ \beta_0 = $' + str(beta_0) +
    # #          r', $ \omega_0 = $' + str(omega_0) + r", $r = $" + str(ratio), r", $ \epsilon = $" + str(epsilon))
    #
    # title = ('$J_1$ for ' + str(spin_number) + ' spins, $h_0=$' + str(h_0) + r', $\beta_0 = $' + str(beta_0) +
    #          r', $\omega_0 = $' + str(omega_0) + r', $r = $' + str(ratio) + r', $\epsilon = $' + str(epsilon))
    # axx.set_title(title)
    # axx.set_xlabel("time")
    # axx.set_ylabel("$J_1$")
    #
    # plotm_t_h(m_0, h, time_arr, spin_number, omega_0, h_0, beta_0, t_final, ratio+1, variance=True, run_number=run_number)

    plt.show()


if __name__ == '__main__':
    main()
