import time

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


@jit
def sinus(t, omega, amp, a_0=0.0, phase=0.0):
    """

    :param t:
    :param omega:
    :param amp:
    :param a_0:
    :param phase: between 0 and 1
    :return:
    """
    return a_0 + amp * np.sin(omega * t + 2*np.pi*phase)


@jit
def RungeKutta(timestep, final_time, initial, omega_0, h_0, beta_0, omega_beta=0.0, eps=0.0, phase=0.0):
    """
    Performs the Runge Kutta 4 algorithm on function dmdt and returns a matrix [t,m(t)]
    :param timestep:
    :param final_time:
    :param initial:
    :param omega_0:
    :param h_0:
    :param beta_0:
    :param omega_beta:
    :param eps:
    :param phase:
    :param g:
    :return:
    """

    amt = int(final_time/timestep)
    time_range = np.linspace(0, final_time, amt)

    y = np.empty(amt)
    y[0] = initial
    for i in range(amt-1):
        k1 = dmdt(y[i], sinus(time_range[i], omega_beta, eps, beta_0, phase),
                  sinus(time_range[i], omega_0, h_0))

        k2 = dmdt(y[i] + timestep*k1/2, sinus(time_range[i]+timestep/2, omega_beta, eps, beta_0, phase),
                  sinus(time_range[i]+timestep/2, omega_0, h_0))

        k3 = dmdt(y[i] + timestep*k2/2, sinus(time_range[i]+timestep/2, omega_beta, eps, beta_0, phase),
                  sinus(time_range[i]+timestep/2, omega_0, h_0))

        k4 = dmdt(y[i] + timestep*k3, sinus(time_range[i]+timestep, omega_beta, eps, beta_0, phase),
                  sinus(time_range[i]+timestep, omega_0, h_0))

        y[i+1] = y[i]+timestep/6*(k1+2*k2+2*k3+k4)

    return [time_range, y]


@jit
def dmdt(m, beta, h):
    return np.sinh(beta * (m + h)) - m * np.cosh(beta * (m + h))


def plotbeta_star():
    h = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6])
    beta = np.array([1, 1.12, 1.29, 1.45, 1.63, 2.05, 2.62, 3.43, 4.7])

    plt.plot(h, beta)
    plt.title(r"$\beta^*$ as function of $h_0$ at $\omega_0=0.1$, $\omega_\beta=0.01$")
    plt.xlabel("$h_0$")
    plt.ylabel(r"$\beta^*$")
    plt.show()


def crit_temp(h_0_arr, omega_0_arr):
    num1 = len(h_0_arr)
    num2 = len(omega_0_arr)

    acc = 0.001 #accuracy

    crit_temp_arr = np.empty([num2, num1])
    lowerestimate = .998
    for h_i in range(num1):
        print(h_i)
        for omega_i in list(reversed(range(num2))):
            dt = 5e-5 / omega_0_arr[omega_i]
            t_final = 2 * np.pi * 2 / omega_0_arr[omega_i]
            l = int(t_final/dt)
            Gotit=False
            while not Gotit:
                m_t = RungeKutta(dt, t_final, 0, omega_0_arr[omega_i], h_0_arr[h_i], lowerestimate)
                if np.sign(m_t[1][(l*6)//8]) == np.sign(m_t[1][-1]):
                    Gotit = True
                    crit_temp_arr[omega_i][h_i] = lowerestimate
                    print('omega=' + str(omega_i))
                    print(lowerestimate)
                    lowerestimate -= 2*acc

                else:
                    lowerestimate += acc
        lowerestimate = crit_temp_arr[0][h_i]
    return crit_temp_arr


def crit_temp_plot():
    # num1 = 40
    # num2 = 40
    # h_0_arr = np.linspace(1e-5, 0.4, num1)
    # omega_0_arr = np.logspace(-4, -0.9, num2)
    # # omega_0_arr = np.array([0.02])
    #
    # timer = time.time()
    # z_arr = crit_temp(h_0_arr, omega_0_arr)
    # print(time.time() - timer)

    # level = MaxNLocator(nbins=30).tick_values(z_arr.min(), z_arr.max())
    #
    # fig2, ax2 = plt.subplots()
    # ax2.set_yscale('log')
    #
    # cf = ax2.contourf(h_0_arr, omega_0_arr, z_arr, cmap="GnBu", levels=level)
    # cbar = fig2.colorbar(cf)
    #
    # ax2.set_title(r'Nonequilibrium Critical Temperature')
    # ax2.set_xlabel(r'$h_0$')
    # ax2.set_ylabel(r'$\omega_0$')
    # cbar.set_label(r'$\beta_{c_2}$')

    num2=1
    num1=60
    h_0_arr = np.linspace(1e-5, 0.6, num=num1)
    omega_0_arr = np.array([0.02])

    z_arr = crit_temp(h_0_arr, omega_0_arr)

    fig2, (axup, axdown) = plt.subplots(1,2, figsize=(7.4,4), layout='tight')
    axup.plot(h_0_arr, z_arr[0])

    num1 = 1
    num2 = 60
    h_0_arr = np.array([0.2])
    omega_0_arr = np.logspace(-4.0, -0.69, num2)


    timer = time.time()
    z_arr2 = crit_temp(h_0_arr, omega_0_arr)
    print(time.time() - timer)
    print(z_arr2)
    axdown.semilogx(omega_0_arr, z_arr2[:,0])

    axup.set_title(r'$\beta_{c_2}(h_0)$ at $\omega_0 = 0.02$')
    axup.set_xlabel(r'$h_0$')
    axup.set_ylabel(r'$\beta_{c_2}$')

    axdown.set_title(r'$\beta_{c_2}(\omega_0)$ at $h_0 = 0.3$')
    axdown.set_xlabel(r'$\omega_0$')
    axdown.set_ylabel(r'$\beta_{c_2}$')


def main():
    crit_temp_plot()
    plt.show()


if __name__ == '__main__':
    main()
