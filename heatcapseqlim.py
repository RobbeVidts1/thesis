import numpy as np
from numba import jit
import matplotlib.pyplot as plt


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
    
    h_t = np.zeros_like(time_range)
    for i in range(final_time/10):
        h_t[i] = 1

    m_0 = np.empty(amt)
    m_0[0] = initial_0
    m_1 = np.empty(amt)
    m_1[0] = initial_1
    J_1 = np.empty(amt)
    for i in range(amt - 1):
        k1 = dmdt(m_0[i], beta_0, h_t[i])

        k2 = dmdt(m_0[i] + timestep * k1 / 2, beta_0, h_t[i])

        k3 = dmdt(m_0[i] + timestep * k2 / 2, beta_0, h_t[i])

        k4 = dmdt(m_0[i] + timestep * k3, beta_0, h_t[i])

        m_0[i + 1] = m_0[i] + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        l1 = dm_1dt(m_0[i], m_1[i], beta_0, omega_beta, h_t[i], time_range[i])

        l2 = dm_1dt(m_0[i] + timestep * k1 / 2, m_1[i] + l1 * timestep / 2, beta_0, omega_beta,
                    h_t[i], time_range[i] + timestep / 2)

        l3 = dm_1dt(m_0[i] + timestep * k2 / 2, m_1[i] + l2 * timestep / 2, beta_0, omega_beta,
                    h_t[i], time_range[i] + timestep / 2)

        l4 = dm_1dt(m_0[i] + timestep * k3, m_1[i] + l3 * timestep, beta_0, omega_beta,
                    h_t[i], time_range[i] + timestep)

        m_1[i + 1] = m_1[i] + timestep / 6 * (l1 + 2 * l2 + 2 * l3 + l4)

        J_1[i] = (m_1[i] * k1 + (m_0[i] + h_t[i]) * l1)

    return [time_range, m_0, m_1, J_1]


def plotJ_1(beta_0, omega_0, omega_beta, h_0, dt, t_final):
    m_t = RungeKutta_split(dt, t_final, initial_0=0.0, initial_1=0.0, omega_0=omega_0, h_0=h_0,
                           beta_0=beta_0, omega_beta=omega_beta)
    q = int(dt*omega_beta*2*np.pi)
    plt.plot(m_t[0][-q:-1], m_t[3][-q:-1] * np.cos(omega_beta * m_t[0][-q:-1]))
    plt.show()


def main():
    omega_0 = 0
    omega_beta = 1e-4
    h_0 = 0

    dt = 2e-2  # A small amount with respect to omega (I will always use omega_beta << omega_0
    t_final = 2 * np.pi * 2 / omega_beta

    step_number = int(t_final / dt)

    amt = 100
    beta = np.linspace(0, 5, amt)
    C = np.empty(amt)

    for i in range(amt):
        m_t = RungeKutta_split(dt, t_final, initial_0=0.0, initial_1=0.0, omega_0=omega_0, h_0=h_0,
                               beta_0=beta[i], omega_beta=omega_beta)

        Jcos = m_t[3][step_number // 2:] * np.cos(omega_beta * m_t[0][step_number // 2:])
        C[i] = beta[i] / np.pi * dt * (sum(Jcos) - (Jcos[0] + Jcos[-1]) / 2)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(beta, C)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("C")
    ax.set_title(r"Heat capacity ($h_0 = 0$, $\omega_\beta = $" + str(omega_beta) + ")")
    # ax.set_ylim(-2.5, 4)
    plt.show()


if __name__ == '__main__':
    main()
