import numpy as np
import random
from numba import jit
import matplotlib.pyplot as plt
import time
import concurrent.futures
from itertools import repeat


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
    m = (2*N_up)/N-1
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

    energy_up = energy(N_up+1, field, N)
    result[0] = dt*(N-N_up)/(1+np.exp(-beta * (original_energy - energy_up)))

    energy_down = energy(N_up-1, field, N)
    result[1] = dt*N_up/(1+np.exp(-beta * (original_energy - energy_down)))

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
def solver(N, field_arr, beta_arr, time_arr, init_steps, dt, exp_steps):
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

    result = np.empty((3, exp_steps))
    result[0] = time_arr[init_steps:]

    start = result[0][0]

    for i in range(init_steps):
        N_up = state_update(N_up, beta_arr[i], field_arr[i], N, dt)

    for i in range(exp_steps):
        N_up = state_update(N_up, beta_arr[init_steps + i], field_arr[init_steps + i], N, dt)
        result[1][i] = (2*N_up-N)/N
        result[2][i] = energy(N_up, field_arr[init_steps + i], N)

    return result

########################################################################
######### Here are the files that write the explanations
#######################################################################

def write_expl_avg_m(omega_0, h_0, beta_0, N, run_number, dt, init_time, name):
    file = open(name+".txt", "w")
    file.write("the data has shape (4,steps)\n"
               "data[0] is the time array\n"
               "data[1] is the field array\n"
               "data[2] is the average magnetization\n"
               "data[3] is the variance on the magnetization\n"
               " variables are:\n"
               "omega_0 = " + str(omega_0) + "\n"
               "h_0 = " + str(h_0) + "\n"
               "beta_0 = " + str(beta_0) + "\n"
               "timestep = " + str(dt) + "\n"
               "N = " + str(N) + "\n"
                "run_number = " + str(run_number) + "\n"
               "init_time = " + str(init_time)
    )
    file.close()


def write_expl_avg_Heatcap(h_0, omega_0, r, N, run_number, dt, name):
    file=open(name, 'w')
    file.write("This data has the average heat capacity for a finite system for a set of beta's\n"
               "data[0] is the beta_array\n"
               "data[1] is the heatcap_array\n"
               "variables are:\n"
               f"omega_0 = {str(omega_0)}\n"
               f"h_0 = {str(h_0)}\n"
               f"timestep = 1/{str(dt)}N\n"
               f"r = {str(r)}\n"
               f"N = {str(N)}"
               f"run_number = {str(run_number)}")


########################################################################
######### Here are the specific examples worked out
#######################################################################

def Trial():
    x = solver(40, 0.7, 0.02, 3.0, 0.02, 0, 1.0, 1.0)

    fig1, ax1 = plt.subplots()
    ax1.plot(x[0], x[1])
    fig2, ax2 = plt.subplots()
    ax2.plot(x[0],x[2])
    plt.show()


def average_magnetization(): 
    # physical parameters
    N = 250

    h_0 = 0.3
    omega_0 = 0.02
    beta_0 = 1.2
    # compute parameters
    exp_time = 1
    init_time = 1

    dt = 1/(4*N)  ## Perhaps we should check if this is indeed sufficient.

    run_number = 5000

    # I have some doubts about np.arange for such long lists of arrays
    # (potential big number + small number error )
    #                                      https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    # creating the arrays for the first one
    time_arr = np.arange(start=-(2 * np.pi * init_time / omega_0), stop=2 * np.pi * exp_time / omega_0, step=dt)
    beta_arr = beta_0 * np.ones_like(time_arr)
    field_arr = h_0 * np.sin(omega_0 * time_arr)

    # finding all the step numbers
    total_steps = len(time_arr)
    init_steps = total_steps // 2 - 10 ## making a guess to make the while loop short
    while time_arr[init_steps + 1] < 0:
        init_steps += 1
    exp_steps = total_steps - init_steps

    avg_m = np.zeros((4, exp_steps))

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        # make sure to remove when using cluster
        func_list = {executor.submit(solver, N, field_arr, beta_arr, time_arr, init_steps, dt, exp_steps)
                for i in range(run_number)}

        for func in concurrent.futures.as_completed(func_list):
            avg_m[2] += func.result()[1]
            avg_m[3] += func.result()[1] ** 2


    avg_m[2] /= run_number
    avg_m[3] /= run_number
    avg_m[0] = solver(N, field_arr, beta_arr, time_arr, init_steps, dt, exp_steps)[0]

    avg_m[1] = field_arr[init_steps:]

    name = "avg_m_" + str(N)

    np.save(name, avg_m)
    write_expl_avg_m(omega_0, h_0, beta_0, N, run_number, dt, init_time, name)
    plt.plot(avg_m[0], avg_m[2])
    plt.show()


def Heatcap(h_0, omega_0, beta_0, r, epsilon, N, dt_int):
    """
    Calculates the heat capacity of a single run given a set of data
    :param h_0:
    :param omega_0:
    :param beta_0:
    :param r:
    :param epsilon:
    :param N:
    :return:
    """
    # additional parameters
    dt = 1/(dt_int*N)
    init_time = 1.0
    exp_time = 1.0 # possibililty to increase exp time to have a smaller fraction of total time be initializing
    # (with 1-1), it's 10 percent
    omega_beta = omega_0/r

    time_arr = np.arange(start=-(2 * np.pi * init_time / omega_0), stop=2 * np.pi * exp_time / omega_beta, step=dt)
    beta_arr = beta_0 * (1+epsilon*np.sin(omega_beta*time_arr))
    field_arr = h_0 * np.sin(omega_0 * time_arr)

    # fiding the step numbers
    total_steps = len(time_arr)
    init_steps = total_steps // (r+1) - 10 ## making a guess to make the while loop short
    while time_arr[init_steps + 1] < 0:
        init_steps += 1
    exp_steps = total_steps - init_steps

    J = np.empty(exp_steps)
    J_base = np.empty_like(J)
    # solving, but only keeping the magnetization

    m = solver(N, field_arr, beta_arr, time_arr, init_steps, dt, exp_steps)
    m_0 = solver(N, field_arr, beta_0 * np.ones_like(time_arr), time_arr, init_steps, dt, exp_steps)
    for i in range(exp_steps):
        # finding an array of heat currents in the base case and the total case (a[init_step+1] is the first exp result)
        J[i] = (m[1][i] + m[1][i - 1] + field_arr[init_steps + i] + field_arr[init_steps + i - 1]) / 2 * \
               (m[1][i] - m[1][i - 1])
        J_base[i] = (m_0[1][i] + m_0[1][i - 1] + field_arr[init_steps + i] + field_arr[init_steps + i-1]) / 2 * \
               (m_0[1][i] - m_0[1][i - 1])
    C = beta_0*np.pi * np.sum(J * np.cos(omega_beta*time_arr[init_steps:])) # there is no dt dependence anymnore,
                                                            # there is one in the denominator from J and one in the
                                                            # numerator from the integration
    return C


def avg_Heatcap_helper(beta_0, N):
    """
    This function is a helper function to avoid exceesive repeats in executor.map() which calculates the
    average heatcap in a single point => Always make sure the variables are the same as in avg_Heatcap
    :param beta_0:
    :return:
    """
    timer = time.time()
    h_0 = 0.3
    omega_0 = 0.02
    r = 12  # Need to check if r=10 and r=20 are equal
    epsilon = 0.2  # Not sure what best value is.
    dt=4

    run_number = 200

    result = 0
    for i in range(run_number):
        result += Heatcap(h_0, omega_0, beta_0, r, epsilon, N, dt)

    print(f'This took {time.time()-timer} seconds')
    return result/run_number


def avg_Heatcap(N):
    h_0 = 0.3
    omega_0 = 0.02
    r = 12 # Need to check if r=10 and r=20 are equal. It seems so. Perhaps use r=12, just to be sure
    epsilon = 0.2 # Not sure what best value is.
    run_number = 20 # 200 is actually quite reasonable already, perhaps 400 or 500 is final goal?
    dt = 4 # use 2,3 or 4?

    transition_estimate = 2.1 # This depends on h_0 and omega_0
    end = 3.8

    # making a beta array with different densities around the transition temp
    beta_density_low = 10 # probably use 20 and 50 to get a total of 130 datapoints
    beta_density_high = 25
    beta_arr = np.append(np.append(np.linspace(0,transition_estimate-0.5, num=int(1.5*beta_density_low),
                                               endpoint=False),
                                   np.linspace(transition_estimate-0.5, transition_estimate+0.5,
                                               num=beta_density_high, endpoint=False)),
                         np.linspace(transition_estimate+0.5, end,
                                     num=int( (end-transition_estimate+0.5) * beta_density_low)))
    print(f'the total number of datapoints is {len(beta_arr)}')

    # performing the calculation in parallel, will need lots of repeats unfortunately
    result_arr = np.empty_like(beta_arr)
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor: # max_workers says how many cpu's to use,
        # make sure to remove when using cluster
        results = executor.map(avg_Heatcap_helper, beta_arr, repeat(N))
        for i,result in enumerate(results):
            result_arr[i] = result
            print(i) # So we know how far along we are

    filename = f'Heatcap_N_{N}_r_{r}'
    np.save(filename+'.npy', np.matrix([beta_arr, result_arr]))
    write_expl_avg_Heatcap(h_0, omega_0, r, N, run_number, dt, filename)


    plt.plot(beta_arr, result_arr)

    plt.show()











def main():
    average_magnetization()
    # avg_Heatcap(10) ## When using cluster, probably make avg_Heatcap(N), and run for all desired N

if __name__ == '__main__':
    main()


