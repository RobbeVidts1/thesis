import time

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from setup import RungeKutta
from basic_units import radians

@jit
def dmdt(m, beta, h):
    return np.sinh(beta * (m + h)) - m * np.cosh(beta * (m + h))


@jit()
def dm_1dt(m_0, m_1, beta_0, omega_beta, h, t):
    """

    :param t:
    :param m_0:
    :param m_1:
    :param beta_0:
    :param omega_beta:
    :param h:
    :return: returns the first order in epsilon of dmdt

    """
    return beta_0 * (np.sin(omega_beta * t) * (m_0 + h) + m_1) * np.cosh(beta_0 * (m_0 + h)) \
        - m_0 * beta_0 * (np.sin(omega_beta * t) * (m_0 + h) + m_1) * np.sinh(beta_0 * (m_0 + h)) \
        - m_1 * np.cosh(beta_0 * (m_0 + h))


@jit()
def RungeKutta_split(timestep, final_time, initial_0, initial_1, omega_0, h_0, beta_0, omega_beta):
    """
    Performs the Runge Kutta 4 algorithm on function dmdt and returns a matrix [t,m(t)]
    :param initial_1:
    :param initial_0:
    :param timestep:
    :param final_time:
    :param omega_0:
    :param h_0:
    :param beta_0:
    :param omega_beta:
    :return:
    """

    amt = int(final_time / timestep)
    time_range = np.linspace(0, final_time, amt)

    m_0 = np.empty(amt)
    m_0[0] = initial_0
    m_1 = np.empty(amt)
    m_1[0] = initial_1
    J_1 = np.empty(amt)
    for i in range(amt - 1):
        k1 = dmdt(m_0[i], beta_0, h_0 * np.sin(omega_0 * time_range[i]))

        k2 = dmdt(m_0[i] + timestep * k1 / 2, beta_0, h_0 * np.sin(omega_0 * (time_range[i] + timestep / 2)))

        k3 = dmdt(m_0[i] + timestep * k2 / 2, beta_0, h_0 * np.sin(omega_0 * (time_range[i] + timestep / 2)))

        k4 = dmdt(m_0[i] + timestep * k3, beta_0, h_0 * np.sin(omega_0 * (time_range[i] + timestep)))

        m_0[i + 1] = m_0[i] + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        l1 = dm_1dt(m_0[i], m_1[i], beta_0, omega_beta, h_0 * np.sin(omega_0 * time_range[i]), time_range[i])

        l2 = dm_1dt(m_0[i] + timestep * k1 / 2, m_1[i] + l1 * timestep / 2, beta_0, omega_beta,
                    h_0 * np.sin(omega_0 * (time_range[i] + timestep / 2)), time_range[i] + timestep / 2)

        l3 = dm_1dt(m_0[i] + timestep * k2 / 2, m_1[i] + l2 * timestep / 2, beta_0, omega_beta,
                    h_0 * np.sin(omega_0 * (time_range[i] + timestep / 2)), time_range[i] + timestep / 2)

        l4 = dm_1dt(m_0[i] + timestep * k3, m_1[i] + l3 * timestep, beta_0, omega_beta,
                    h_0 * np.sin(omega_0 * (time_range[i] + timestep)), time_range[i] + timestep)

        m_1[i + 1] = m_1[i] + timestep / 6 * (l1 + 2 * l2 + 2 * l3 + l4)

        J_1[i] = (m_1[i] * k1 + (m_0[i] + h_0 * np.sin(omega_0 * time_range[i])) * l1)

    return [time_range, m_0, m_1, J_1]


def plotJ_0():
    h_0 = 0.3
    omega_0 = 0.02

    beta_1 = 1.0
    beta_2 = 1.8
    dt = 5e-5
    t_final = 2 * np.pi/omega_0 * 2

    m_1 = RungeKutta(dt, t_final, 0, omega_0, h_0, beta_1)
    m_2 = RungeKutta(dt, t_final, 0, omega_0, h_0, beta_2)

    h_arr = h_0 * np.sin(omega_0 * m_1[0])

    l = len(h_arr)

    J_1 = np.empty_like(h_arr)
    J_2 = np.empty_like(h_arr)

    for i in range(len(h_arr)):
        J_1[i] = (m_1[1][i] + h_arr[i]) * dmdt(m_1[1][i], beta_1, h_arr[i])
        J_2[i] = (m_2[1][i] + h_arr[i]) * dmdt(m_2[1][i], beta_2, h_arr[i])
    t_rescaled = (m_1[0][l//2:]-m_1[0][l//2])*omega_0*radians

    fig, (axup, axdown) = plt.subplots(2, 1, layout='constrained', sharex=True)

    axup.plot(t_rescaled, m_1[1][l//2:], label=r'$\beta=1$', xunits=radians)
    axup.plot(t_rescaled, m_2[1][l//2:], label=r'$\beta=1.8$', xunits=radians)
    axup.plot(t_rescaled, h_arr[l//2:], color='black', alpha=0.5, label='$h$', xunits=radians)

    axdown.plot(t_rescaled, J_1[l//2:], xunits=radians)
    axdown.plot(t_rescaled, J_2[l//2:], xunits=radians)

    axup.set_title('Magnetization')
    axup.set_ylabel('$m$')
    axup.legend(loc='upper right')

    axdown.set_title('Heat Current')
    axdown.set_ylabel('$I$')
    axdown.set_xlabel(r'$\omega_0t$')

    fig.suptitle('Heat Current for CW Magnet with $h_0 = $' + str(h_0) + r', $\omega_0 = $' + str(omega_0))
    axdown.grid(visible=True, axis='both', linewidth=0.5, alpha=0.3, color='k')
    axup.grid(visible=True, axis='both', linewidth=0.5, alpha=0.3, color='k')
    plt.show()



def plotJ_1():
    beta_0 = 1.25
    h_0 = 0.3
    omega_0 = 0.02
    ratio = 10
    omega_beta = omega_0/ratio

    dt=5e-5
    t_final = 2*np.pi*2/omega_beta
    m_t = RungeKutta_split(dt, t_final, initial_0=0.0, initial_1=0.0, omega_0=omega_0, h_0=h_0,
                           beta_0=beta_0, omega_beta=omega_beta)
    q = len(m_t[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), layout='constrained')

    t_rescaled = (m_t[0][q//(ratio+1):]-m_t[0][q//(ratio+1)])*omega_0*radians

    ax.plot(t_rescaled, m_t[3][q//(ratio+1):], xunits=radians)
    ax.set_title(r'Excess Heat Current ($\beta=1.25$, $h_0=0.3$, $\omega_0=0.02$)')
    ax.set_xlabel(r'$\omega_0 t$')
    ax.set_ylabel(r'$I_1$')
    ax.grid(visible=True, axis='both', linewidth=0.5, alpha=0.3, color='k')
    plt.show()


def plotbeta_star():
    h = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6])
    beta = np.array([1, 1.12, 1.29, 1.45, 1.63, 2.05, 2.62, 3.43, 4.7])

    plt.plot(h, beta)
    plt.title(r"$\beta^*$ as function of $h_0$ at $\omega_0=0.1$, $\omega_\beta=0.01$")
    plt.xlabel("$h_0$")
    plt.ylabel(r"$\beta^*$")
    plt.show()


def plotm1(beta_0, omega_0, omega_beta, h_0, dt, t_final):
    m_t = RungeKutta_split(dt, t_final, initial_0=0.0, initial_1=0.0, omega_0=omega_0, h_0=h_0,
                           beta_0=beta_0, omega_beta=omega_beta)
    q = int(t_final/dt/2)
    epsilon = 0.05

    plt.plot(m_t[0][-q:-1], epsilon*m_t[2][-q:-1], label=r"$\epsilon m_1$", alpha=0.7)
    plt.plot(m_t[0][-q:-1], m_t[1][-q:-1], alpha=0.5, c="red", label="$m_0$", ls="--")
    plt.plot(m_t[0][-q:-1], m_t[1][-q:-1] + epsilon*m_t[2][-q:-1], c="green", alpha=0.5, label="$m$", ls="--")

    plt.legend(loc='lower left')

    plt.show()


def plotmultipleomega():
    omega_array = [ 0.2, 0.1, .05,.01]
    h_0 = 0.3

    fig, ax = plt.subplots(layout="constrained")
    for j in range(len(omega_array)):
        omega_0 = omega_array[j]
        omega_beta = omega_0/20

        dt = 1e-4 / omega_0  # A small amount with respect to omega (I will always use omega_beta << omega_0
        t_final = 2 * np.pi * 2 / omega_beta
        step_number = int(t_final / dt)

        amt = 250
        beta = np.linspace(0.01, 3, amt)
        C = np.empty(len(beta))

        for i in range(len(beta)):
            m_t = RungeKutta_split(dt, t_final, initial_0=0.0, initial_1=0.0, omega_0=omega_0, h_0=h_0,
                                   beta_0=beta[i], omega_beta=omega_beta)

            Jcos = m_t[3][step_number // 2:] * np.cos(omega_beta * m_t[0][step_number // 2:])
            C[i] = beta[i] / np.pi * dt * (sum(Jcos) - (Jcos[0] + Jcos[-1]) / 2)

        ax.plot(beta, C, label=r"$\omega_0=$" + str(omega_0))

    ax.set_xlabel(r"$J\beta$")
    ax.set_ylabel("$C/k_B$")
    ax.set_title(r"Heat Capacity ($h_0 = $" + str(h_0) + ")")
    ax.set_ylim(-1.5, 1.5)
    ax.grid(visible=True, axis='both', linewidth=0.5, alpha=0.3, color='k')
    ax.legend()


def crit_temp(h_0_arr, omega_0_arr):
    num1 = len(h_0_arr)
    num2 = len(omega_0_arr)

    crit_temp_arr = np.empty([num2, num1])
    lowerestimate = .998
    for h_i in range(num1):
        print(h_i)
        for omega_i in range(num2):
            dt = 5e-5 / omega_0_arr[omega_i]
            t_final = 2 * np.pi * 2 / omega_0_arr[omega_i]
            l = int(t_final/dt)
            Gotit=False
            while Gotit==False:
                m_t = RungeKutta(dt, t_final, 0, omega_0_arr[omega_i], h_0_arr[h_i], lowerestimate)
                if np.sign(m_t[1][(l*6)//8]) == np.sign(m_t[1][-1]):
                    Gotit = True
                    crit_temp_arr[omega_i][h_i] = lowerestimate
                    lowerestimate -= 0.2
                    print('omega=' + str(omega_i))

                else:
                    lowerestimate += 0.005
        lowerestimate = crit_temp_arr[0][h_i]
    return crit_temp_arr


def crit_temp_plot():
    num1 = 30
    num2 = 30
    h_0_arr = np.linspace(1e-5, 0.5, num1)
    omega_0_arr = np.logspace(-4.0, -0.9, num2)
    # omega_0_arr = np.array([0.02])

    timer = time.time()
    z_arr = crit_temp(h_0_arr, omega_0_arr)
    print(time.time() - timer)
    print(z_arr)

    level = MaxNLocator(nbins=30).tick_values(z_arr.min(), z_arr.max())

    fig2, ax2 = plt.subplots()
    ax2.set_yscale('log')

    cf = ax2.contourf(h_0_arr, omega_0_arr, z_arr, cmap="GnBu", levels=level)
    cbar = fig2.colorbar(cf)


    ax2.set_title(r'Nonequilibrium Critical Temperature')
    ax2.set_xlabel(r'$h_0$')
    ax2.set_ylabel(r'$\omega_0$')
    cbar.set_label(r'$\beta_{c_2}$')

    # fig2, (axup, axdown) = plt.subplots(1,2, layout='tight')
    # axup.plot(h_0_arr, z_arr[0])
    #
    # num1 = 1
    # num2 = 50
    # h_0_arr = np.array([0.3])
    # omega_0_arr = np.logspace(-4.0, -0.69, num2)
    #
    #
    # timer = time.time()
    # z_arr2 = crit_temp(h_0_arr, omega_0_arr)
    # print(time.time() - timer)
    # print(z_arr2)
    # axdown.semilogx(omega_0_arr, z_arr2[:,0])
    #
    # axup.set_title(r'$\beta_{c_2}(h_0)$ at $\omega_0 = 0.02$')
    # axup.set_xlabel(r'$h_0$')
    # axup.set_ylabel(r'$\beta_{c_2}$')
    #
    # axdown.set_title(r'$\beta_{c_2}(\omega_0)$ at $h_0 = 0.3$')
    # axdown.set_xlabel(r'$\omega_0$')
    # axdown.set_ylabel(r'$\beta_{c_2}$')


def compare_r():
    omega_0 = 0.02
    h_0 = 0.3
    r_arr = [3, 10, 20, 50]

    dt = 5e-5 / omega_0  # A small amount with respect to omega (I will always use omega_beta << omega_0

    beta_num = 100

    beta_arr = np.linspace(0,3.5, beta_num)

    C = np.empty_like(beta_arr)

    fig, ax = plt.subplots(1, 1, layout='tight')

    for r_i in range(len(r_arr)):
        omega_beta = omega_0/r_arr[r_i]
        t_final = 2 * np.pi * 2 / omega_beta
        step_number = int(t_final / dt)
        for beta_i in range(beta_num):
            m_t = RungeKutta_split(dt, t_final, initial_0=-0.1, initial_1=-0.1, omega_0=omega_0, h_0=h_0,
                           beta_0=beta_arr[beta_i], omega_beta=omega_beta)

            Jcos = m_t[3][step_number // 2:] * np.cos(omega_beta * m_t[0][step_number // 2:])
            C[beta_i] = beta_arr[beta_i] / np.pi * dt * (sum(Jcos) - (Jcos[0] + Jcos[-1]) / 2)
        # row = r_i % 2
        # column = r_i // 2
        # axs[row, column].plot(beta_arr, C)
        # axs[row, column].set_title()
        ax.plot(beta_arr, C, label=('$r = $' + str(r_arr[r_i])))

    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('$C$')
    ax.set_title('CW Heat Capacity for Different $r$')
    ax.legend()
    ax.set_ylim(-1.5, 1.5)

def plots_C():
    omega_0 = 0.02
    r = 10
    h_arr = [1e-4, 0.05, 0.1, 0.2]

    dt = 5e-5 / omega_0  # A small amount with respect to omega (I will always use omega_beta << omega_0

    beta_num = 100

    beta_arr = np.linspace(0,2.7, beta_num)

    C = np.empty_like(beta_arr)

    fig, axs = plt.subplots(2, 2, figsize=(6.4*1.2, 4.8*1.2), layout='tight', sharex=True, sharey=True)

    for h_i in range(len(h_arr)):
        omega_beta = omega_0/r
        t_final = 2 * np.pi * 2 / omega_beta
        step_number = int(t_final / dt)
        for beta_i in range(beta_num):
            m_t = RungeKutta_split(dt, t_final, initial_0=-0.1, initial_1=-0.1, omega_0=omega_0, h_0=h_arr[h_i],
                           beta_0=beta_arr[beta_i], omega_beta=omega_beta)

            Jcos = m_t[3][step_number // 2:] * np.cos(omega_beta * m_t[0][step_number // 2:])
            C[beta_i] = beta_arr[beta_i] / np.pi * dt * (sum(Jcos) - (Jcos[0] + Jcos[-1]) / 2)
        row = h_i // 2
        column = h_i % 2
        axs[row, column].plot(beta_arr, C)
        axs[row, column].set_title('$h_0=$'+str(h_arr[h_i]))
        axs[row,column].grid(visible=True, axis='both', linewidth=0.5, alpha=0.3, color='k')

    axs[1,0].set_xlabel(r'$J\beta$')
    axs[1,0].set_ylabel('$C/k_B$')
    axs[0,0].set_ylabel('$C/k_B$')
    axs[1,1].set_xlabel(r'$J\beta$')
    fig.suptitle(r'CW Heat Capacity for Different $h_0$ ($\omega_0 = $'+str(omega_0)+')')
    axs[0,0].set_ylim(-1.5, 1.5)



def main():
    # omega_0 = 2e-2
    # omega_beta = omega_0/20
    # h_0 = 0.3
    #
    # dt = 5e-5 / omega_0  # A small amount with respect to omega (I will always use omega_beta << omega_0
    # t_final = 2 * np.pi * 2 / omega_beta
    #
    # step_number = int(t_final / dt)
    #
    # amt = 1000
    # beta = np.linspace(0.001, 3.5, amt)
    # C = np.empty(len(beta))
    #
    # for i in range(len(beta)):
    #     m_t = RungeKutta_split(dt, t_final, initial_0=0.0, initial_1=0.0, omega_0=omega_0, h_0=h_0,
    #                            beta_0=beta[i], omega_beta=omega_beta)
    #
    #     Jcos = m_t[3][step_number // 2:] * np.cos(omega_beta * m_t[0][step_number // 2:])
    #     C[i] = beta[i] / np.pi * dt * (sum(Jcos) - (Jcos[0] + Jcos[-1]) / 2)
    #
    # np.savetxt("C_inf.csv", C, delimiter=",")
    # fig, ax = plt.subplots(figsize=(2*6.4, 2*4.8), layout="constrained")
    # ax.plot(beta, C)
    # ax.set_xlabel(r"$J\beta$")
    # ax.set_ylabel("$C/k_B$")
    # ax.set_title(r"Heat Capacity for $\omega_0 = $" + str(omega_0) + r", $\omega_\beta = $" + str(omega_beta)
    #              + r", $h_0 = $" + str(h_0))
    # ax.set_ylim(-1.5, 1.0)
    # ax.grid(visible=True, axis='both', linewidth=0.5, alpha=0.3, color='k')
    #
    # betaspecial = [0.8, 1.2, 1.6, 2.0, 2.19]
    #
    #
    # # for i in range(len(beta)):
    # #     print(beta[i], C[i])
    #
    # # betaspecial = [1.55, 1.6, 1.625, 1.65, 1.675, 1.75]
    # C_special = np.empty_like(betaspecial)
    # for i in range(len(betaspecial)):
    #     m_t = RungeKutta_split(dt, t_final, initial_0=0.0, initial_1=0.0, omega_0=omega_0, h_0=h_0,
    #                            beta_0=betaspecial[i], omega_beta=omega_beta)
    #
    #     Jcos = m_t[3][step_number // 2:] * np.cos(omega_beta * m_t[0][step_number // 2:])
    #     C_special[i] = betaspecial[i] / np.pi * dt * (sum(Jcos) - (Jcos[0] + Jcos[-1]) / 2)
    #
    # for i in range(len(betaspecial)):
    #     print(i, betaspecial[i], C_special[i])
    # ax.scatter(betaspecial, C_special, c='red', alpha=0.8)
    #

    plotmultipleomega()
    # plotm1(1.4, omega_0, omega_beta, h_0, dt, t_final)
    # plots_C()
    plt.show()

    # omega_0 = 0.02
    # r = 3
    # h_0 = 1e-2
    #
    # dt = 5e-5 / omega_0  # A small amount with respect to omega (I will always use omega_beta << omega_0
    #
    # beta_num = 100
    #
    # beta_arr = np.linspace(1.049, 1.053, beta_num)
    #
    # C = np.empty_like(beta_arr)
    # omega_beta = omega_0 / r
    # t_final = 2 * np.pi * 2 / omega_beta
    # step_number = int(t_final / dt)
    # for beta_i in range(beta_num):
    #     m_t = RungeKutta_split(dt, t_final, initial_0=-0.1, initial_1=-0.1, omega_0=omega_0, h_0=h_0,
    #                            beta_0=beta_arr[beta_i], omega_beta=omega_beta)
    #
    #     Jcos = m_t[3][step_number // 2:] * np.cos(omega_beta * m_t[0][step_number // 2:])
    #     C[beta_i] = beta_arr[beta_i] / np.pi * dt * (sum(Jcos) - (Jcos[0] + Jcos[-1]) / 2)
    # l = np.argmax(C)
    # print(beta_arr[l])
    # print(C[l])
    # plotJ_0()
    # plotJ_1()

if __name__ == '__main__':
    main()
