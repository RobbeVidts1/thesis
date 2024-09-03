import numpy as np
import random
from numba import jit
import matplotlib.pyplot as plt
from basic_units import radians

from micro_solver import average_magnetization

def average_magnetization_plot():
    """
    makes both the m(t) and m(h) plots for the average magnetization. it also adds the macroscopic result
    :return:
    """
    m_50 = np.load("avg_m_50.npy")
    m_100 = np.load("avg_m_100.npy")
    m_macro = np.load("simple_solve.npy")

    ## In here are the things that need to be read from the txt file
    omega_0 = 0.02
    h_0 = 0.3
    beta_0 = 1.3

    ## I have to put time still in units of radians
    time_50 = m_50[0] * omega_0 * radians
    time_100 = m_100[0] * omega_0 * radians
    time_macro = m_macro[0] * omega_0 * radians


    # Setting lay-out variables
    dpi_set = 200.0
    figsize_set = [6.4, 4.8]
    plt.rcParams.update({
        "axes.labelsize": 15,  # size of labels
        "grid.alpha": .5,  # visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 15,
        "lines.linewidth": 1.5,
        "xtick.labelsize": 15,  # size of ticks
        "ytick.labelsize": 15
    })

    fig, ax = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
    ## use larger dpi when making final figure ?
    ax.plot(time_50, m_50[2], xunits=radians, label=r"$N = 50$")
    ax.plot(time_100, m_100[2], xunits=radians, label=r"$N = 100$")
    ax.plot(time_macro, m_macro[2], xunits=radians, label=r"macroscopic")

    ax.set_xlabel("$\omega_0 t$")
    ax.set_ylabel("$m_t$")

    plt.grid(True)
    ax.legend(loc="upper right")

    ## Here comes the parametrized version of the plot



    fig_alt, ax_alt = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
    ## use larger dpi when making final figure
    ax_alt.plot(m_50[1], m_50[2], label=r"$N = 50 $")
    ax_alt.plot(m_100[1], m_100[2], label=r"$N = 100$")
    ax_alt.plot(m_macro[1], m_macro[2], label=r"macroscopic")

    ax_alt.set_xlabel("$h_t$")
    ax_alt.set_ylabel("$m_t$")

    plt.grid(True)
    ax_alt.legend(loc="lower right")

    plt.show()


def heatcap_plot():
    data = np.load("Heatcap_N_50_r_10.npy")
    # data_2 = np.load("Heatcap_N_50_r_20.npy")

    dpi_set = 200.0
    figsize_set = [6.4, 4.8]
    plt.rcParams.update({
        "axes.labelsize": 15,  # size of labels
        "grid.alpha": .5,  # visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 15,
        "lines.linewidth": 1.5,
        "xtick.labelsize": 15,  # size of ticks
        "ytick.labelsize": 15
    })

    fig, ax = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
    ax.plot(data[0], data[1])
    # ax.plot(data_2[0], data_2[1])

    plt.show()




def main():
    # average_magnetization_plot()
    heatcap_plot()

if __name__ == '__main__':
    main()
