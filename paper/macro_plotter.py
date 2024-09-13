from bisect import bisect_left

import numpy as np
import matplotlib.pyplot as plt
from holoviews.plotting.bokeh.styles import font_size

from basic_units import radians


# plt.rcParams['text.usetex'] = True


def first_hyst_fig_plot():
    """
    Makes both th m(t) figure and the m(h) figure. Does not save them automatically, but this can be changed
    :return:
    """
    data = np.load("Data/fig_trans.npy")

    ## In here are the things that need to be read from the txt file
    omega_0 = 0.02
    h_0 = 0.3
    beta_0 = np.array([1.8, 2.1, 2.2, 2.5])

    ## I have to put time still in units of radians
    time = data[0] * omega_0 * radians
    h_t = h_0*np.sin(data[0]*omega_0)

    # Setting lay-out variables
    dpi_set = 200.0
    figsize_set = [6.4,4.8]
    plt.rcParams.update({
        "axes.labelsize": 15, #size of labels
        "axes.linewidth": 1.2,
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 15,
        "lines.linewidth": 2,
        "xtick.labelsize": 15,  #size of ticks
        "ytick.labelsize": 15,
    })

    fig, ax=plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
            ## use larger dpi when making final figure ?
    ax.plot(time, data[1], xunits=radians, label=r"$\beta = $" + str(beta_0[0]))
    ax.plot(time, data[2], xunits=radians, label=r"$\beta = $" + str(beta_0[1]))
    ax.plot(time, data[3], xunits=radians, label=r"$\beta = $" + str(beta_0[2]))
    ax.plot(time, data[4], xunits=radians, label=r"$\beta = $" + str(beta_0[3]))
    ax.plot(time, h_t, xunits=radians, c='grey', alpha=0.8, ls='--')

    ax.set_xlabel(r"$\omega_0 t$")
    ax.set_ylabel("$m_t$")

    plt.grid(True)
    ax.legend()

    ## Here comes the parametrized version of the plot
    h_t = h_0 * np.sin(omega_0*data[0])

    fig_alt, ax_alt = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
            ## use larger dpi when making final figure
    ax_alt.plot(h_t, data[1], label=r"$\beta = $" + str(beta_0[0]))
    ax_alt.plot(h_t, data[2], label=r"$\beta = $" + str(beta_0[1]))
    ax_alt.plot(h_t, data[3], label=r"$\beta = $" + str(beta_0[2]))
    ax_alt.plot(h_t, data[4], label=r"$\beta = $" + str(beta_0[3]))


    ax_alt.set_xlabel("$h_t$")
    ax_alt.set_ylabel("$m_t$")

    plt.grid(True)
    ax_alt.legend()

    plt.show()


def infl_omega_0_plot():
    """
    Makes the m(h) figures for changing omega_0 in both the warm and the cold case.
        Does not save them automatically, but this can be changed
    :return:
    """

    #warm
    dataset = np.load("infl_omega_0_warm.npz")
    data = dataset["data"]
    stat_data = dataset["stat_data"]
    dataset.close()

    ## In here are the things that need to be read from the txt file
    omega_0 = np.array([0.002, 0.02, 0.1])
    h_0 = 0.7
    beta_0_warm = 0.85
    beta_0_cold = 2

    # Setting lay-out variables
    dpi_set = 200.0
    figsize_set = [6.4,4.8]
    plt.rcParams.update({
        "axes.labelsize": 15, #size of labels
        "axes.linewidth": 1.2,
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        "xtick.labelsize": 15,  #size of ticks
        "ytick.labelsize": 15,
    })

    ## Here comes the parametrized version of the plot
    h_t = h_0 * np.sin(data[0])

    fig_alt_warm, ax_alt_warm = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
            ## use larger dpi when making final figure
    ax_alt_warm.plot(h_t, data[1], label=r"$\omega_0 = $" + str(omega_0[0]))
    ax_alt_warm.plot(h_t, data[2], label=r"$\omega_0 = $" + str(omega_0[1]))
    ax_alt_warm.plot(h_t, data[3], label=r"$\omega_0 = $" + str(omega_0[2]))

    ax_alt_warm.plot(stat_data[0], stat_data[1], label="stationary solution")

    ax_alt_warm.set_xlabel("$h_t$")
    ax_alt_warm.set_ylabel("$m_t$")

    plt.grid(True)
    ax_alt_warm.legend(loc="upper left")

    #cold
    dataset = np.load("infl_omega_0_cold.npz")
    data = dataset["data"]
    stat_data = dataset["stat_data"]
    dataset.close()

    ## In here are the things that need to be read from the txt file
    omega_0 = np.array([0.002, 0.02, 0.1])
    h_0 = 0.7
    beta_0_warm = 0.85
    beta_0_cold = 2

    # Setting lay-out variables
    dpi_set = 200.0
    figsize_set = [6.4,4.8]
    plt.rcParams.update({
        "axes.labelsize": 15, #size of labels
        "axes.linewidth": 1.2,
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        "xtick.labelsize": 15,  #size of ticks
        "ytick.labelsize": 15,
    })

    ## Here comes the parametrized version of the plot
    h_t = h_0 * np.sin(data[0])

    fig_alt_cold, ax_alt_cold = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
            ## use larger dpi when making final figure
    ax_alt_cold.plot(h_t, data[1], label=r"$\omega_0 = $" + str(omega_0[0]))
    ax_alt_cold.plot(h_t, data[2], label=r"$\omega_0 = $" + str(omega_0[1]))
    ax_alt_cold.plot(h_t, data[3], label=r"$\omega_0 = $" + str(omega_0[2]))

    ax_alt_cold.plot(stat_data[0], stat_data[1], label="stationary solution")

    ax_alt_cold.set_xlabel("$h_t$")
    ax_alt_cold.set_ylabel("$m_t$")

    plt.grid(True)
    ax_alt_cold.legend(loc="upper left")


    plt.show()


def unbounded_heatcap_plot():
    # physical parameters
    omega_0 = 0.02
    omega_beta=0.001

    # load the data
    data = np.load("Data/Heatcap_unbounded.npy")

    # plotparameters
    dpi_set = 200.0
    figsize_set = [6.4,4.8]
    plt.rcParams.update({
        "axes.labelsize": 15, #size of labels
        "axes.linewidth": 1.2,
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 15,
        "lines.linewidth": 2,
        "xtick.labelsize": 15,  #size of ticks
        "ytick.labelsize": 15,
    })

    # plotting
    fig, ax = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')

    plt.plot(data[0], data[1], label=r"$h_0 = 0.0001$")
    plt.plot(data[0], data[2], label=r"$h_0 = 0.01$")
    plt.plot(data[0], data[3], label=r"$h_0 = 0.08$")
    plt.plot(data[0], data[4], label=r"$h_0 = 0.16$")
    plt.plot(data[0], data[5], label=r"$h_0 = 0.3$")

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("$C$")

    ax.set_xlim(0,2.6)
    ax.set_ylim(-2,2.2)

    plt.grid(True)
    ax.legend()

    plt.show()


def heatcap_explan_fig():
    data08 = np.load("Data/simple_solve_beta_8.0.npy")
    data12 = np.load("Data/simple_solve_beta_12.0.npy")
    data16 = np.load("Data/simple_solve_beta_16.0.npy")
    data20 = np.load("Data/simple_solve_beta_20.0.npy")
    data22 = np.load("Data/simple_solve_beta_22.0.npy")

    heatcap = np.load("Data/Heatcap_unbounded.npy")

    dpi_set = 100.0
    figsize_set = [6.4,4.8]

    h = 0.3*np.sin(0.02*data08[0])

    prespecials = [0.8, 1.2, 1.6, 2.0, 2.2]

    specials=np.empty((2,len(prespecials)))

    for i in range(len(prespecials)):
        pos = bisect_left(heatcap[0], prespecials[i])
        specials[0][i] = heatcap[0][pos]
        specials[1][i] = heatcap[5][pos]


    dpi_set = 200.0
    figsize_set = [6.4,4.8]
    plt.rcParams.update({
        "axes.labelsize": 15, #size of labels
        "axes.linewidth": 1.2,
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 15,
        "lines.linewidth": 2,
        "xtick.labelsize": 15,  #size of ticks
        "ytick.labelsize": 15,
         'axes.titlesize': 25
    })

    fig_C, ax_C = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
    ax_C.plot(heatcap[0], heatcap[5])
    ax_C.scatter(specials[0], specials[1], color='red')
    ax_C.set_xlabel(r"$\beta$")
    ax_C.set_ylabel(r"$C$")
    ax_C.set_xlim(-0.1, 3.1)
    ax_C.set_ylim(-1.8, 1.1)
    plt.grid(True)

    plt.rcParams.update({
        "axes.labelsize": 25, #size of labels
        "axes.linewidth": 1.5,
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .6,
        "legend.fontsize": 15,
        "lines.linewidth": 3,
        "xtick.labelsize": 0.01,
        "xtick.labelcolor": 'white',
        "ytick.labelsize": 0.01,
        "ytick.labelcolor": 'white',
        'axes.titlesize': 30
    })


    plt.savefig("explan_C.png", transparent=True)

    fig_08, ax_08 = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
    ax_08.plot(h, data08[2])
    ax_08.set_title(r"$\beta=0.8$", y=0, loc="right")
    ax_08.set_xlabel(r"$h_t$")
    ax_08.set_ylabel(r"$m_t$")
    ax_08.set_ylim(-1.05, 1.05)
    ax_08.set_xlim(-0.315, 0.315)
    plt.grid(True)

    plt.savefig("explan_08.png", transparent=True)

    fig_12, ax_12 = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
    ax_12.plot(h, data12[2])
    ax_12.set_title(r"$\beta=1.2$", y=0, loc="right")
    ax_12.set_xlabel(r"$h_t$")
    ax_12.set_ylabel(r"$m_t$")
    ax_12.set_ylim(-1.05, 1.05)
    ax_12.set_xlim(-0.315, 0.315)
    plt.grid(True)

    plt.savefig("explan_12.png", transparent=True)

    fig_16, ax_16 = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
    ax_16.plot(h, data16[2])
    ax_16.set_title(r"$\beta=1.6$", y=0, loc="right")
    ax_16.set_xlabel(r"$h_t$")
    ax_16.set_ylabel(r"$m_t$")
    ax_16.set_ylim(-1.05, 1.05)
    ax_16.set_xlim(-0.315, 0.315)
    plt.grid(True)

    plt.savefig("explan_16.png", transparent=True)

    fig_20, ax_20 = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
    ax_20.plot(h, data20[2])
    ax_20.set_title(r"$\beta=2.0$", y=0, loc="right")
    ax_20.set_xlabel(r"$h_t$")
    ax_20.set_ylabel(r"$m_t$")
    ax_20.set_ylim(-1.05, 1.05)
    ax_20.set_xlim(-0.315, 0.315)
    plt.grid(True)

    plt.savefig("explan_20.png", transparent=True)

    fig_22, ax_22 = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
    ax_22.plot(h, data22[2])
    ax_22.set_title(r"$\beta=2.2$", y=0, loc="right")
    ax_22.set_xlabel(r"$h_t$")
    ax_22.set_ylabel(r"$m_t$")
    ax_22.set_ylim(-1.05, 1.05)
    ax_22.set_xlim(-0.315, 0.315)
    plt.grid(True)

    plt.savefig("explan_22.png", transparent=True)
    plt.show()


def bdd_unbdd_fig():
    data_bdd = np.load("Data/simple_solve_beta_16.0_bdd.npy")
    data_unbdd = np.load("Data/simple_solve_beta_16.0.npy")

    heatcap_unbdd = np.load("Data/Heatcap_unbounded_03.npy")
    heatcap_bdd = np.load("Data/Heatcap_bounded_03.npy")

    # Setting lay-out variables
    dpi_set = 200.0
    figsize_set = [6.4,4.8]
    plt.rcParams.update({
        "axes.labelsize": 15, #size of labels
        "axes.linewidth": 1.2,
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 15,
        "lines.linewidth": 2,
        "xtick.labelsize": 15,  #size of ticks
        "ytick.labelsize": 15,
    })

    fig, ax=plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
            ## use larger dpi when making final figure ?
    ax.plot(data_unbdd[1], data_unbdd[2], label=r"unbounded")
    ax.plot(data_bdd[1], data_bdd[2], label=r"bounded")


    ax.set_xlabel(r"$h_t$")
    ax.set_ylabel("$m_t$")

    plt.grid(True)
    ax.legend()

    fig_C, ax_C = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
            ## use larger dpi when making final figure ?
    ax_C.plot(heatcap_unbdd[0], heatcap_unbdd[1], label=r"unbounded")
    ax_C.plot(heatcap_bdd[0], heatcap_bdd[1], label=r"bounded")


    ax_C.set_xlabel(r"$\beta$")
    ax_C.set_ylabel("$C$")
    ax_C.set_ylim(-2.0, 1.0)
    ax_C.set_xlim(-0.05, 3.1)

    plt.grid(True)
    ax_C.legend()

    plt.show()


def critical_temp_fig():
    data_h = np.load("critical_temp_omega_0.02.npy")
    data_omega = np.load("Data/critical_temp_h_0.3.npy")

    dpi_set = 200.0
    figsize_set = [6.4,4.8]
    plt.rcParams.update({
        "axes.labelsize": 15, #size of labels
        "axes.linewidth": 1.2,
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 15,
        "lines.linewidth": 2,
        "xtick.labelsize": 15,  #size of ticks
        "ytick.labelsize": 15,
    })

    fig1, ax1 = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
    plt.grid(True)
    ax1.plot(data_h[0], data_h[1])
    ax1.set_xlim(-0.01, 0.5)
    ax1.set_ylim(0.9,4.1)
    ax1.set_xlabel("$h_0$")
    ax1.set_ylabel(r"$\beta_c$")


    fig2, ax2 = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
    plt.grid(True)
    ax2.plot(data_omega[0], data_omega[1])
    ax2.set_xlabel("$\omega_0$")
    ax2.set_ylabel(r"$\beta_c$")
    plt.show()









def main():
    # first_hyst_fig_plot()
    # infl_omega_0_plot()
    # unbounded_heatcap_plot()
    # heatcap_explan_fig()
    bdd_unbdd_fig()
    # critical_temp_fig()

if __name__ == '__main__':
    main()