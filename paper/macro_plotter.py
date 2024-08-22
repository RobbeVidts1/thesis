import numpy as np
import matplotlib.pyplot as plt
from basic_units import radians

plt.rcParams['text.usetex'] = True


def first_hyst_fig_plot():
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


def main():
    first_hyst_fig_plot()


if __name__ == '__main__':
    main()