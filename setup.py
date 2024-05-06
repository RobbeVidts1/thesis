import numpy as np
from numba import jit


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
def dmdt(m, beta, h, nu=1.0, g=0.0):
    return nu * (np.sinh(beta * (m + h + (g * m ** 3))) - m * np.cosh(beta * (m + h + (g * m ** 3))))


@jit
def RungeKutta(timestep, final_time, initial, omega_0, h_0, beta_0, omega_beta=0.0, eps=0.0, phase=0.0, g=0.0):
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
                  sinus(time_range[i], omega_0, h_0), g=g)

        k2 = dmdt(y[i] + timestep*k1/2, sinus(time_range[i]+timestep/2, omega_beta, eps, beta_0, phase),
                  sinus(time_range[i]+timestep/2, omega_0, h_0), g=g)

        k3 = dmdt(y[i] + timestep*k2/2, sinus(time_range[i]+timestep/2, omega_beta, eps, beta_0, phase),
                  sinus(time_range[i]+timestep/2, omega_0, h_0), g=g)

        k4 = dmdt(y[i] + timestep*k3, sinus(time_range[i]+timestep, omega_beta, eps, beta_0, phase),
                  sinus(time_range[i]+timestep, omega_0, h_0), g=g)

        y[i+1] = y[i]+timestep/6*(k1+2*k2+2*k3+k4)

    return [time_range, y]


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


