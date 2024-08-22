import numpy as np
from numba import jit
import scipy.optimize as opt
import matplotlib.pyplot as plt

################################################################
#%% A load of functions used in calculations        ###############
################################################################

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
def dmdt(m, beta, h):
    """

    :param m: current magnetization
    :param beta: current temperature
    :param h: current field strength
    :return: the derivative of the magnetization
    """
    return np.sinh(beta * (m + h)) - m * np.cosh(beta * (m + h))


@jit()
def dm_1dt(m_0, m_1, beta_0, omega_beta, h, t):
    """
    Finds the first order (in epsilon) perturbation of dmdt
    :param t: time
    :param m_0: zero_order magnetization
    :param m_1: first order magnetization
    :param beta_0: temp
    :param omega_beta:
    :param h:
    :return: returns the first order in epsilon of dmdt

    """
    return beta_0 * (np.sin(omega_beta * t) * (m_0 + h) + m_1) * np.cosh(beta_0 * (m_0 + h)) \
        - m_0 * beta_0 * (np.sin(omega_beta * t) * (m_0 + h) + m_1) * np.sinh(beta_0 * (m_0 + h)) \
        - m_1 * np.cosh(beta_0 * (m_0 + h))


@jit
def RungeKutta_simple(timestep, final_time, initial, omega_0, h_0, beta_0, omega_beta=0.0, eps=0.0, phase=0.0):
    """
    Performs the Runge Kutta 4 algorithm on function dmdt and returns a matrix [t,m(t)]
    :param timestep: Delta t between two time points
    :param final_time: final_time
    :param initial: initial magnetization
    :param omega_0:
    :param h_0:
    :param beta_0:
    :param omega_beta:
    :param eps:
    :param phase:
    :return:
    """

    amt = int(final_time/timestep)
    time_range = np.linspace(0, final_time, amt)

    m_t = np.empty(amt)
    m_t[0] = initial
    for i in range(amt-1):
        k1 = dmdt(m_t[i], sinus(time_range[i], omega_beta, eps, beta_0, phase),
                  sinus(time_range[i], omega_0, h_0))

        k2 = dmdt(m_t[i] + timestep*k1/2, sinus(time_range[i]+timestep/2, omega_beta, eps, beta_0, phase),
                  sinus(time_range[i]+timestep/2, omega_0, h_0))

        k3 = dmdt(m_t[i] + timestep*k2/2, sinus(time_range[i]+timestep/2, omega_beta, eps, beta_0, phase),
                  sinus(time_range[i]+timestep/2, omega_0, h_0))

        k4 = dmdt(m_t[i] + timestep*k3, sinus(time_range[i]+timestep, omega_beta, eps, beta_0, phase),
                  sinus(time_range[i]+timestep, omega_0, h_0))

        m_t[i+1] = m_t[i]+timestep/6*(k1+2*k2+2*k3+k4)

    return [time_range, m_t]


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


def stationary(field_list, beta):
    """
    :param field_list: an ordered list
        containing the field strengths over which you want the stationary solution
    :param beta: float
        the inverse temperature
    :return: array(2*len(field_list)x 2)
        result[:,0] contains the solutions to dmdt=0
        result[:,1] contains the field strength twice
    """
    estimate = -1.0
    result = np.empty((2,2*len(field_list)) )

    # rising field
    for i in range(len(field_list)):
        inter = opt.root(dmdt, estimate, args=(beta, field_list[i]))

        while not inter.success:
            estimate += 0.1
            inter = opt.root(dmdt, estimate, args=(beta, field_list[i]))

        result[1][i], result[0][i] = inter.x, field_list[i]
        estimate = result[1][i]

    # lowering field
    for i in range(len(field_list)):
        inter = opt.root(dmdt, estimate, args=(beta, field_list[-(i+1)]))

        while not inter.success:
            estimate -= 0.1
            inter = opt.root(dmdt, estimate, args=(beta, field_list[-(i+1)]))

        result[1][len(field_list)+i], result[0][len(field_list)+i] = inter.x, field_list[-(i+1)]
        estimate = result[1][len(field_list)+i]

    return result
#%%

################################################################
#%% Functions related to saving the data       ###############
################################################################

def write_expl_first_figure(omega_0, h_0, beta_0, omega_beta, m_init, dt, init_time):
    file = open("fig_trans.txt", "w")
    file.write("the data has shape (5,steps)\n"
               "data[0] is the time array\n"
               "all other rows are magnetizations\n"
               "data[1]:beta_0 = "+str(beta_0[0]) +
                "\n data[2]:beta_0 = " + str(beta_0[1]) +
                "\n data[3]:beta_0 = " + str(beta_0[2]) +
                "\n data[4]:beta_0 = " + str(beta_0[3]) +
               "\n other variables are:"
               "\n omega_0 = " + str(omega_0) +
               "\n h_0 = " + str(h_0) +
               "\n omega_beta = " + str(omega_beta) +
               "\n m_init = " + str(m_init) +
               "\n timestep = " + str(dt) +
               "\n init_time = " + str(init_time)
    )
    file.close()


def write_expl_omega_0_fig(omega_0, h_0, beta_0, omega_beta, m_init, dt, init_time):
    file = open("infl_omega_0_warm.txt", "w")
    file.write("the data comtains two arrays, data (4,steps) and stat_data (2, steps) \n"
               "data[0] is the time array\n"
               "all other rows are magnetizations\n"
               "data[1]:omega_0 = " + str(omega_0[0]) +
               "\n data[2]:omega_0 = " + str(omega_0[1]) +
               "\n data[3]:omega_0 = " + str(omega_0[2]) +
               "\n other variables are:"
               "\n beta_0 = " + str(beta_0) +
               "\n h_0 = " + str(h_0) +
               "\n omega_beta = " + str(omega_beta) +
               "\n m_init = " + str(m_init) +
               "\n timestep = " + str(dt) +
               "\n init_time = " + str(init_time) +
               "\n the stat_data contains stationary solutions"
               "\n stat_data[0] is a field list"
               "\n stat_data[1] is the stationary solution at that field"
               )
    file.close()

################################################################
#%% The calculations and the main        ###############
################################################################

def first_hyst_fig():
    ### parameters ###
    #physical parameters
    omega_0 = 2e-2
    h_0 = 0.3
    beta_0 = [1.8, 2.1, 2.2, 2.5]
    m_init = 0 #initial value of magnetization
    init_time = 1 #number of periods before measurement
    measure_time = 1 #number of periods for measurement

    #calculation parameters
    dt = 5e-5 / omega_0  # A small amount with respect to omega (I will always use omega_beta << omega_0
    t_final = (init_time + measure_time) * np.pi * 2 / omega_0
    step_number = int(t_final / dt)

    for i in range(len(beta_0)):
        m_t = RungeKutta_simple(dt, t_final, m_init, omega_0, h_0, beta_0[i])
        if i==0:
            result = np.empty((5, len(m_t[0][(step_number * measure_time) // (init_time+measure_time):])))
                            ## keeping only the measurement periods
        result[i+1] = m_t[1][step_number // 2:]

    result[0] = (m_t[0][step_number // 2:] - m_t[0][(step_number * measure_time) // (init_time+measure_time)])

    # plt.plot(result[0], result[1])
    # plt.show()

    np.save("fig_trans.npy", result)

    write_expl_first_figure(omega_0, h_0, beta_0, np.nan, m_init, dt, 1)


def influence_omega_0_warm():
    ### parameters ###
    #physical parameters
    omega_0 = [0.002, 0.02, 0.1]
    h_0 = 0.7
    beta_0 = 0.85
    m_init = 0 #initial value of magnetization
    init_time = 1 #number of periods before measurement
    measure_time = 1 #number of periods for measurement

    for i in range(len(omega_0)):
        # calculation parameters now depend on omega_0
        dt = 5e-5 / omega_0[i]  # A small amount with respect to omega (I will always use omega_beta << omega_0
        t_final = (init_time + measure_time) * np.pi * 2 / omega_0[i]
        step_number = int(t_final / dt)

        # do calculation
        m_t = RungeKutta_simple(dt, t_final, m_init, omega_0[i], h_0, beta_0)
        if i==0:
            result = np.empty((5, len(m_t[0][(step_number * measure_time) // (init_time+measure_time):]))) ## making the results array

        result[i+1] = m_t[1][(step_number * measure_time) // (init_time+measure_time):] ## keeping only the measurement periods

    # adding the time (in radians) to the results array
    time = (m_t[0][(step_number * measure_time) // (init_time+measure_time):] -
            m_t[0][(step_number * measure_time) // (init_time+measure_time)])
    result[0]= omega_0[-1] * time

    #### Now I should add the stationary one
    h = np.linspace(-h_0, h_0, 1000)

    stat_result = stationary(h, beta_0)

    # plt.plot(result[0], result[1])
    # plt.show()

    np.savez("infl_omega_0_warm.npz", data=result, stat_data=stat_result)

    write_expl_omega_0_fig(omega_0, h_0, beta_0, np.nan, m_init, dt, 1)


def influence_omega_0_cold():
    ### parameters ###
    #physical parameters
    omega_0 = [0.002, 0.02, 0.1]
    h_0 = 0.7
    beta_0 = 1.5
    m_init = 0 #initial value of magnetization
    init_time = 1 #number of periods before measurement
    measure_time = 1 #number of periods for measurement

    for i in range(len(omega_0)):
        # calculation parameters now depend on omega_0
        dt = 5e-5 / omega_0[i]  # A small amount with respect to omega (I will always use omega_beta << omega_0
        t_final = (init_time + measure_time) * np.pi * 2 / omega_0[i]
        step_number = int(t_final / dt)

        # do calculation
        m_t = RungeKutta_simple(dt, t_final, m_init, omega_0[i], h_0, beta_0)
        if i==0:
            result = np.empty((5, len(m_t[0][(step_number * measure_time) // (init_time+measure_time):]))) ## making the results array

        result[i+1] = m_t[1][(step_number * measure_time) // (init_time+measure_time):] ## keeping only the measurement periods

    # adding the time (in radians) to the results array
    time = (m_t[0][(step_number * measure_time) // (init_time+measure_time):] -
            m_t[0][(step_number * measure_time) // (init_time+measure_time)])
    result[0]= omega_0[-1] * time

    #### Now I should add the stationary one
    h = np.linspace(-h_0, h_0, 1000)

    stat_result = stationary(h, beta_0)

    # plt.plot(result[0], result[1])
    # plt.show()

    np.savez("infl_omega_0_cold.npz", data=result, stat_data=stat_result)

    write_expl_omega_0_fig(omega_0, h_0, beta_0, np.nan, m_init, dt, 1)


def main():
    # first_hyst_fig()
    influence_omega_0_warm()
    influence_omega_0_cold()







if __name__ == '__main__':
    main()
