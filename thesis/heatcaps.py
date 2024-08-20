import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import csv


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


def plotJ_1(beta_0, omega_0, omega_beta, h_0, dt, t_final):
    m_t = RungeKutta_split(dt, t_final, initial_0=0.0, initial_1=0.0, omega_0=omega_0, h_0=h_0,
                           beta_0=beta_0, omega_beta=omega_beta)
    q = int(dt*omega_beta*2*np.pi)
    plt.plot(m_t[0][-q:-1], m_t[3][-q:-1] * np.cos(omega_beta * m_t[0][-q:-1]))
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
    omega_array = [0.5, 0.2, 1e-1, 1e-2,]
    h_0 = 0.2

    fig, ax = plt.subplots(layout="constrained")
    for j in range(len(omega_array)):
        omega_0 = omega_array[j]
        omega_beta = omega_0/10

        dt = 1e-4 / omega_0  # A small amount with respect to omega (I will always use omega_beta << omega_0
        t_final = 2 * np.pi * 2 / omega_beta
        step_number = int(t_final / dt)

        amt = 50
        beta = np.linspace(0.3, 3, amt)
        C = np.empty(len(beta))

        for i in range(len(beta)):
            m_t = RungeKutta_split(dt, t_final, initial_0=0.0, initial_1=0.0, omega_0=omega_0, h_0=h_0,
                                   beta_0=beta[i], omega_beta=omega_beta)

            Jcos = m_t[3][step_number // 2:] * np.cos(omega_beta * m_t[0][step_number // 2:])
            C[i] = beta[i] / np.pi * dt * (sum(Jcos) - (Jcos[0] + Jcos[-1]) / 2)

        ax.plot(beta, C, label="omega_0=" + str(omega_0))

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("C")
    ax.set_title(r"Heat capacity ($\frac{\omega_0}{\omega_\beta} = 10$, $h_0 = $" + str(h_0) + ")")
    ax.set_ylim(-1.5, 2)
    ax.legend()


def main():
    omega_0 = 2e-2
    omega_beta = omega_0/10
    h_0 = 0.3

    dt = 5e-5 / omega_0  # A small amount with respect to omega (I will always use omega_beta << omega_0
    t_final = 2 * np.pi * 2 / omega_beta

    step_number = int(t_final / dt)

    amt = 200
    beta = np.linspace(0.005, 3.5, amt)
    C = np.empty(len(beta))

    for i in range(len(beta)):
        m_t = RungeKutta_split(dt, t_final, initial_0=0.0, initial_1=0.0, omega_0=omega_0, h_0=h_0,
                               beta_0=beta[i], omega_beta=omega_beta)

        Jcos = m_t[3][step_number // 2:] * np.cos(omega_beta * m_t[0][step_number // 2:])
        C[i] = beta[i] / np.pi * dt * (sum(Jcos) - (Jcos[0] + Jcos[-1]) / 2)

    fig, ax = plt.subplots(layout="constrained")
    ax.axhline(y=0, color='black', alpha=0.3, ls='--')
    ax.plot(beta, C)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("$C$")
    ax.set_title(r"Heat capacity for $\omega_0 = $" + str(omega_0) + r", $\omega_\beta = $" + str(omega_beta)
                 + r", $h_0 = $" + str(h_0))
    ax.set_ylim(-1.5, 1.0)

    # for i in range(len(beta)):
    #     print(beta[i], C[i])

    # betaspecial = [1.55, 1.6, 1.625, 1.65, 1.675, 1.75]
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


    # plotmultipleomega()
    # plotm1(1.4, omega_0, omega_beta, h_0, dt, t_final)

    plt.show()

    # plotbeta_star()



if __name__ == '__main__':
    main()
