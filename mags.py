import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import scipy.optimize as opt
import setup
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

"""Variable names: 
    m = instantaneous magnetization of the system (function of time)
    h = instantaneous external field (function of t)
        h depends on omega_0 and amp
    beta = instantaneous inverse temperature (function of time)
        depends on omega_beta and epsilon_beta
    nu = thing that makes time dimensionless
    g = amount of m^4/4 we have in psi
        

"""


@jit
def inversetemp(t, omega_beta, epsilon, beta_0):
    return beta_0 + epsilon * np.sin(omega_beta * t)


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
    result = np.empty((2*len(field_list), 2))

    # rising field
    for i in range(len(field_list)):
        inter = opt.root(setup.dmdt, estimate, args=(beta, field_list[i]))

        while not inter.success:
            estimate += 0.1
            inter = opt.root(setup.dmdt, estimate, args=(beta, field_list[i]))

        result[i][0], result[i][1] = inter.x, field_list[i]
        estimate = result[i][0]

    # lowering field
    for i in range(len(field_list)):
        inter = opt.root(setup.dmdt, estimate, args=(beta, field_list[-(i+1)]))

        while not inter.success:
            estimate -= 0.1
            inter = opt.root(setup.dmdt, estimate, args=(beta, field_list[-(i+1)]))

        result[len(field_list)+i][0], result[len(field_list)+i][1] = inter.x, field_list[-(i+1)]
        estimate = result[len(field_list)+i][0]

    return result


@jit
def calc_m(amp, omega_0, beta_0, dt, t_final, omega_beta=1.0, epsilon=0.0, phase=0.0, g=0.0):
         # We want to run a few full cycles

    return setup.RungeKutta(dt, t_final, 0.0, omega_0, amp, beta_0, omega_beta, epsilon, phase=phase, g=g)


def mosaicplot(tm, omega_0, h_0, omega_beta, epsilon, beta_0):
    """
    Creates a figure with 3 plots (h(t) and m(t)), beta(t) and m(h)
    :param tm:
    :param omega_0:
    :param h_0:
    :param omega_beta:
    :param epsilon:
    :param beta_0:
    :return:
    """
    fig0, axs = plt.subplot_mosaic([['hmt', 'hmt', 'hm', 'hm'], ['betat', 'betat', 'hm', 'hm']])

    h = setup.sinus(tm[0], omega_0, h_0)

    axs['hmt'].plot(tm[0], h, label="h")  # The first plot is of m and h as a function of time
    axs['hmt'].plot(tm[0], tm[1], label="m")

    beta = setup.sinus(tm[0], omega_beta, epsilon, beta_0)

    axs['betat'].plot(tm[0], beta)

    axs['hmt'].set_ylabel("field strength")
    axs['hmt'].set_xlabel("t")
    axs['hmt'].set_title("fields")
    axs['hmt'].legend(loc='lower left')

    axs['betat'].set_xlabel("t")
    axs['betat'].set_ylabel(r"$\beta$")

    axs['hm'].plot(h, tm[1])
    axs['hm'].set_xlabel('h')
    axs['hm'].set_ylabel('m')

    fig0.suptitle("$h_0=$" + str(h_0) + r"$, \omega_0=$" + str(omega_0) + r", $\beta_0=$" + str(beta_0) + r', omega='
                  + str(omega_beta) + r", $\epsilon=$" + str(epsilon))

    plt.tight_layout()
    plt.show()


def plotm_h():
    epsilon = 0
    h_0 = 0.3
    beta = 2.1
    omega_0 = 0.02
    # omega_beta = round(omega_0*(1 + 0.2*np.random.random()), 6)
    omega_beta = 0.1
    # phi = np.random.random()
    phi = 0
    colorscaling = False
    t_final = 2*np.pi/omega_0*2

    m_of_t = calc_m(h_0, omega_0, beta, 1e-4/omega_0, t_final, omega_beta, epsilon, phase=phi)
    delta_beta = setup.sinus(m_of_t[0][:], omega_beta, epsilon, phase=phi)
    h = setup.sinus(m_of_t[0][:], omega_0, h_0)

    l = len(m_of_t[0])

    fig, (axleft, axright) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout='constrained')
    title = (r"Magnetization around $\beta_{c_2}$ for "
             r"$ \omega_0 = $" + str(omega_0) +
             # r", $\omega_\beta = $" + str(omega_beta) +
             # r", $\beta = $" + str(beta) +
             r", $h_0 = $" + str(h_0)
             # + r", $\epsilon = $" + str(epsilon)
             )
    fig.suptitle(title)

    axleft.plot(m_of_t[0][l // 2:], m_of_t[1][l // 2:], label=r"$m$ at $\beta=2.1$")


    # axleft.plot(m_of_t[0][l // 2:], 5 * delta_beta[l // 2:], alpha=0.6, label=r"$5*\Delta \beta$")
    axleft.axhline(y=0, color='black', linestyle='--', lw=0.4, alpha=0.4)
    axleft.set_xlabel("$t$")

    res2 = calc_m(h_0, omega_0, beta, 1e-5/omega_0, omega_beta, 0)

    axright.set_xlim(-h_0 - 0.05, h_0 + 0.05)
    axright.set_ylim(-1.05, 1.05)

    axright.plot(h[l // 2:], m_of_t[1][l // 2:], label=r"$m$ at $\beta=2.1$")

    m_of_t = calc_m(h_0, omega_0, beta+0.1, 1e-4 / omega_0, t_final, omega_beta, epsilon, phase=phi)

    axleft.plot(m_of_t[0][l // 2:], m_of_t[1][l // 2:], label=r"$m$ at $\beta=2.2$", color='r')

    axright.plot(h[l // 2:], m_of_t[1][l // 2:], label=r"$m$ at $\beta=2.2$", color='r')

    axleft.plot(m_of_t[0][l // 2:], h[l // 2:], alpha=0.6, label="$h$")


    if colorscaling:
        points = np.array([h[l // 2:], m_of_t[1][l // 2:]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(m_of_t[0][2 * l // 12], m_of_t[0][-1])
        lc = LineCollection(segs, cmap='jet', norm=norm, alpha=0.6, lw=0.85)
        lc.set_array(m_of_t[0][l // 2:])

        axright.add_collection(lc)
    # else:
        # axright.plot(h[l // 2:], m_of_t[1][l // 2:])

    # axright.plot(h[l // 2:], res2[1][l // 2:], alpha=0.7, label=r"$\epsilon = 0$", color='black', lw=0.75)
    # axright.legend(loc="lower right")
    axright.axhline(y=0, color='black', linestyle='--', lw=0.4, alpha=0.4)
    axright.axvline(x=0, color='black', linestyle='--', lw=0.4, alpha=0.4)

    axleft.legend(loc='lower left')
    axright.legend(loc='lower right')

    axright.set_xlabel("h")
    axright.set_ylabel("m")

    plt.show()


def plotcompare_lin_method():
    epsilon = 0.02
    h_0 = 0.2
    beta_0 = 1.4
    omega_0 = 0.02
    omega_beta = omega_0/4

    dt = 8e-5 / omega_0  # A small amount with respect to omega
    t_final = 2 * np.pi * 2 / omega_beta

    colorscaling = True

    m_of_t = setup.RungeKutta(dt, t_final, 0.0, omega_0, h_0, beta_0, omega_beta, epsilon)
    delta_beta = setup.sinus(m_of_t[0][:], omega_beta, epsilon)
    h = setup.sinus(m_of_t[0][:], omega_0, h_0)
    m1_of_t = setup.RungeKutta_split(dt, t_final, 0, 0, omega_0, h_0, beta_0, omega_beta)
    m_comp = m1_of_t[1] + epsilon*m1_of_t[2]

    m_of_t_big = setup.RungeKutta(dt/8, t_final, 0.0, omega_0, h_0, beta_0, omega_beta, epsilon)
    m1_of_t_big = setup.RungeKutta_split(dt/8, t_final, 0, 0, omega_0, h_0, beta_0, omega_beta)

    fig, ax = plt.subplots()
    ax.plot(m_of_t[0], m1_of_t[1] + epsilon*m1_of_t[2] - m_of_t[1], c="blue", label=(r"$dt = $"+str(dt) ))
    ax.plot(m_of_t_big[0], m1_of_t_big[1] + epsilon*m1_of_t_big[2] - m_of_t_big[1], ls=":", c='orange',
            label=(r"$dt = $" + str(dt/8)))
    ax.set_title(r"$\Delta m$ for $\omega_0 = $" + str(omega_0) + r", $\beta_0 = $" + str(beta_0) + r", $h_0 = $"
                 + str(h_0) + r", $\epsilon = $" + str(epsilon))
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\Delta m$")
    ax.legend(loc='upper right')

    l = len(m_of_t[0])

    fig, (axleft, axright) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    title = (r"magnetization for $ \omega_0 = $" + str(omega_0) + r", $\omega_\beta = $" + str(omega_beta) +
             r", $\beta = $" + str(beta_0) + r", $h_0 = $" + str(h_0) + r", $\epsilon = $" + str(epsilon))
    fig.suptitle(title)

    axleft.plot(m_of_t[0][l // 2:], m_of_t[1][l // 2:], label="m (full)", alpha=0.6)
    axleft.plot(m_of_t[0][l // 2:], m_comp[l // 2:], label="m (1st order)", ls="dotted")
    axleft.plot(m_of_t[0][l // 2:], 5 * delta_beta[l // 2:], alpha=0.6, label=r"$5*\Delta \beta$")
    axleft.plot(m_of_t[0][l // 2:], h[l // 2:], alpha=0.6, label="h")
    axleft.axhline(y=0, color='black', linestyle='--', lw=0.4, alpha=0.4)
    axleft.legend(loc='lower left')
    axleft.set_xlabel("t")
    axleft.set_ylim(-1.2, 1.2)

    axright.set_xlim(-h_0 - 0.05, h_0 + 0.05)
    axright.set_ylim(-1.2, 1.2)

    if colorscaling:
        points = np.array([h[l // 2:], m_of_t[1][l // 2:]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(m_of_t[0][l // 2], m_of_t[0][-1])
        lc = LineCollection(segs, cmap='jet', norm=norm, alpha=0.6, lw=0.85)
        lc.set_array(m_of_t[0][l // 2:])

        axright.add_collection(lc)

        axright.plot(h[l // 2:], m_comp[l // 2:], lw=0.75, ls=":", color="black", alpha=0.8)

        axright.add_collection(lc)
    else:
        axright.plot(h[l // 2:], m_of_t[1][l // 2:])

    # axright.plot(h[l // 2:], res2[1][l // 2:], alpha=0.7, label=r"$\epsilon = 0$", color='black', lw=0.75)
    # axright.legend(loc="lower right")
    axright.set_xlabel("h")
    axright.set_ylabel("m")

    proxy1 = Line2D([0, 1], [0, 1], color="black", ls="-", lw=0.85)
    proxy2 = Line2D([0, 1], [0, 1], color="black", ls=":", lw=0.75, alpha=0.8)
    axright.legend([proxy1, proxy2], ['m (full)', 'm (1st order)'], loc='lower right')

    plt.show()


    def plotJ():
        epsilon = 0
        h_0 = 0.5
        beta_0 = 1.4
        omega_0 = 0.02
        omega_beta = omega_0

        m_of_t = calc_m(h_0, omega_0, beta_0, 1e-4 / omega_0, t_final, omega_beta, epsilon)


def main():
    # field_list = np.linspace(-0.8, 0.8, 500)
    # for beta_0 in [0.5, 0.75, 1.0, 1.25, 1.5]:
    #     exact = stationary(field_list, beta_0)
    #     label = r'$\beta = $' + str(beta_0)
    #     plt.plot(exact[:, 1], exact[:, 0], label=label)
    #
    # plt.xlabel('$h$')
    # plt.ylabel(r'$m$')
    # plt.title(r'Stationary solution of the Curie-Weiss magnet')
    #
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    plotm_h()




if __name__ == '__main__':
    main()
