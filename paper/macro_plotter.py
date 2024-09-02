import numpy as np
import matplotlib.pyplot as plt
from basic_units import radians

from paper.macro_solver import HeatCap_unbounded

# plt.rcParams['text.usetex'] = True


def first_hyst_fig_plot():
    """
    Makes both th m(t) figure and the m(h) figure. Does not save them automatically, but this can be changed
    :return:
    """
    data = np.load("fig_trans.npy")

    ## In here are the things that need to be read from the txt file
    omega_0 = 0.02
    h_0 = 0.3
    beta_0 = np.array([1.8, 2.1, 2.2, 2.5])

    ## I have to put time still in units of radians
    time=data[0] * omega_0 * radians

    # Setting lay-out variables
    dpi_set = 100.0
    figsize_set = [6.4,4.8]
    plt.rcParams.update({
        "axes.labelsize": 15,    #size of labels
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 15,
        "lines.linewidth": 1,
        "xtick.labelsize": 15,  #size of ticks
        "ytick.labelsize": 15
    })

    fig, ax=plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
            ## use larger dpi when making final figure ?
    ax.plot(time, data[1], xunits=radians, label=r"$\beta_0 = $" + str(beta_0[0]))
    ax.plot(time, data[2], xunits=radians, label=r"$\beta_0 = $" + str(beta_0[1]))
    ax.plot(time, data[3], xunits=radians, label=r"$\beta_0 = $" + str(beta_0[2]))
    ax.plot(time, data[4], xunits=radians, label=r"$\beta_0 = $" + str(beta_0[3]))

    ax.set_xlabel("$\omega_0 t$")
    ax.set_ylabel("$m$")

    plt.grid(True)
    ax.legend()

    ## Here comes the parametrized version of the plot
    h_t = h_0 * np.sin(omega_0*data[0])

    fig_alt, ax_alt = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
            ## use larger dpi when making final figure
    ax_alt.plot(h_t, data[1], label=r"$\beta_0 = $" + str(beta_0[0]))
    ax_alt.plot(h_t, data[2], label=r"$\beta_0 = $" + str(beta_0[1]))
    ax_alt.plot(h_t, data[3], label=r"$\beta_0 = $" + str(beta_0[2]))
    ax_alt.plot(h_t, data[4], label=r"$\beta_0 = $" + str(beta_0[3]))


    ax_alt.set_xlabel("$h(t)$")
    ax_alt.set_ylabel("$m(t)$")

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
    dpi_set = 100.0
    figsize_set = [6.4,4.8]
    plt.rcParams.update({
        "axes.labelsize": 15,    #size of labels
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 15,
        "lines.linewidth": 1,
        "xtick.labelsize": 15,  #size of ticks
        "ytick.labelsize": 15
    })

    ## Here comes the parametrized version of the plot
    h_t = h_0 * np.sin(data[0])

    fig_alt_warm, ax_alt_warm = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')
            ## use larger dpi when making final figure
    ax_alt_warm.plot(h_t, data[1], label=r"$\omega_0 = $" + str(omega_0[0]))
    ax_alt_warm.plot(h_t, data[2], label=r"$\omega_0 = $" + str(omega_0[1]))
    ax_alt_warm.plot(h_t, data[3], label=r"$\omega_0 = $" + str(omega_0[2]))

    ax_alt_warm.plot(stat_data[0], stat_data[1], label="stationary solution")

    ax_alt_warm.set_xlabel("$h(t)$")
    ax_alt_warm.set_ylabel("$m(t)$")

    plt.grid(True)
    ax_alt_warm.legend()

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
    dpi_set = 100.0
    figsize_set = [6.4,4.8]
    plt.rcParams.update({
        "axes.labelsize": 15,    #size of labels
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 15,
        "legend.loc": "upper left",
        "lines.linewidth": 1,
        "xtick.labelsize": 15,  #size of ticks
        "ytick.labelsize": 15
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
    ax_alt_cold.legend()


    plt.show()


def unbounded_heatcap_plot():
    # physical parameters
    omega_0 = 0.02
    omega_beta=0.001

    # load the data
    data = np.load("Heatcap_unbounded.npy")

    # plotparameters
    dpi_set = 100.0
    figsize_set = [6.4,4.8]
    plt.rcParams.update({
        "axes.labelsize": 15,    #size of labels
        "grid.alpha": .5,       #visibility of grid
        "grid.color": "grey",
        "grid.linestyle": "-",
        "grid.linewidth": .4,
        "legend.fontsize": 15,
        "legend.loc": "upper left",
        "lines.linewidth": 1.5,
        "xtick.labelsize": 15,  #size of ticks
        "ytick.labelsize": 15
    })

    # plotting
    fig, ax = plt.subplots(figsize=figsize_set, dpi=dpi_set, layout='constrained')

    plt.plot(data[0], data[1], label=r"$h_0 = 0.01$")
    plt.plot(data[0], data[2], label=r"$h_0 = 0.08$")
    plt.plot(data[0], data[3], label=r"$h_0 = 0.16$")
    plt.plot(data[0], data[4], label=r"$h_0 = 0.3$")

    ax.set_xlabel(r"$\beta_0$")
    ax.set_ylabel("$C$")

    ax.set_xlim(0,2.6)
    ax.set_ylim(-2,2.2)

    plt.grid(True)
    ax.legend()

    plt.show()



def main():
    # first_hyst_fig_plot()
    # infl_omega_0_plot()
    unbounded_heatcap_plot()

if __name__ == '__main__':
    main()