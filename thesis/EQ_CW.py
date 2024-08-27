import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib.ticker import MaxNLocator
import scipy.optimize as opt



@jit()
def C_2_formula(h, beta):
    return beta**2 * np.exp(beta)*(
            (4*h**2 + 1)*np.cosh(2*beta*h) + 4*h*np.sinh(2*beta*h) + 4*h**2 * np.exp(beta) )/ (
            1+np.exp(beta)*np.cosh(2*beta*h))**2


def C_2():
    h = np.linspace(-0,0.8, 5000)
    beta = np.linspace(0.001, 7, 5000)
    beta, h = np.meshgrid(beta, h)
    z = C_2_formula(h, beta)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(beta, h, z, cmap="jet", linewidth=0)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # ax.set_title("Heat capacity for equilibrium Curie Weiss model for $N=2$")
    # ax.set_xlabel(r"$J\beta$")
    # ax.set_ylabel(r"$h$")
    # ax.set_zlabel(r"$C/k_B$")
    #
    level = MaxNLocator(nbins=25).tick_values(z.min(), z.max())

    fig2, ax2 = plt.subplots(layout='constrained')

    cf = ax2.contourf(beta, h, z, cmap="GnBu", levels=level)
    cbar = fig2.colorbar(cf)
    cbar.set_label(r"$C/k_B$")

    ax2.set_title("Heat Capacity for Equilibrium CW magnet with $N=2$")
    ax2.set_xlabel(r"$J\beta$")
    ax2.set_ylabel(r"$h$")

    plt.show()


def m_2_formula(h, beta):
    return np.exp(beta)*np.cosh(2*beta*h)/(1+np.exp(beta)*np.cosh(2*beta*h))


def m_2():
    h = np.linspace(-0, 0.8, 5000)
    beta = np.linspace(0.001, 5, 5000)
    beta, h = np.meshgrid(beta, h)
    z = m_2_formula(h, beta)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(beta, h, z, cmap="jet", linewidth=0)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # ax.set_title("Heat capacity for equilibrium Curie Weiss model for $N=2$")
    # ax.set_xlabel(r"$J\beta$")
    # ax.set_ylabel(r"$h$")
    # ax.set_zlabel(r"$C/k_B$")
    #
    level = MaxNLocator(nbins=25).tick_values(z.min(), z.max())

    fig2, ax2 = plt.subplots()

    cf = ax2.contourf(beta, h, z, cmap="GnBu", levels=level)
    cbar = fig2.colorbar(cf)
    cbar.set_label(r"$m$")

    ax2.set_title(r"$\langle m\rangle$ for equilibrium Curie Weiss model for $N=2$")
    ax2.set_xlabel(r"$J\beta$")
    ax2.set_ylabel(r"$h$")

    fig2.tight_layout()

    plt.show()


def E_2_formula(h, beta):
    return -np.exp(beta)*(np.cosh(2*beta*h) + 2*h*np.sinh(2*beta*h))/(1+np.exp(beta)*np.cosh(2*beta*h))


def E_2():
    h = np.linspace(-0.75, 0.75, 5000)
    beta = np.linspace(0.001, 10, 5000)
    beta, h = np.meshgrid(beta, h)
    z = E_2_formula(h, beta)

    level = MaxNLocator(nbins=20).tick_values(z.min(), z.max())

    fig2, ax2 = plt.subplots()

    cf = ax2.contourf(beta, h, z, cmap="jet", levels=level)
    cbar = fig2.colorbar(cf)
    cbar.set_label(r"$\langle E \rangle /k_B$")

    ax2.set_title("Average energy for equilibrium Curie Weiss model for $N=2$")
    ax2.set_xlabel(r"$J\beta$")
    ax2.set_ylabel(r"$h$")

    fig2.tight_layout()

    plt.show()


def free_energy_formula(beta, m, h, N):
    return np.exp(-N*(-beta*(m**2/2+h*m) + (1-m)/2*np.log((1-m)/2) + (1+m)/2*np.log((1+m)/2)))


def free_energy():
    x_arr = np.linspace(-0.99, 0.99, 5000)
    y_arr = np.linspace(-0.5, 0.5, 5000)
    beta = 2.0

    x_arr, y_arr = np.meshgrid(x_arr, y_arr)
    z_arr = free_energy_formula(beta, x_arr, y_arr)

    level = MaxNLocator(nbins=30).tick_values(z_arr.min(), z_arr.max())

    fig2, ax2 = plt.subplots()

    cf = ax2.contour(x_arr, y_arr, z_arr, cmap="jet", levels=level)
    cbar = fig2.colorbar(cf)
    ax2.axhline(color='black', ls="--")

    ax2.set_title('free energy function')



    plt.show()


def m_stat(x, beta, h):
    return x-np.tanh(beta*(x+h))


def C_infty():
    N = 500
    h_arr = np.linspace(0, 0.9, num=N)
    beta_arr = np.linspace(0.05, 2, num=N)

    C_arr = np.empty([N,N])

    for i in range(N):
        for j in range(N):
            m_s = opt.root(m_stat, 1.0, args=(beta_arr[i], h_arr[j])).x
            C_arr[j][i] = ((1-m_s*m_s)*(beta_arr[i]*(m_s + h_arr[j]))**2 /
                           (1-beta_arr[i] + beta_arr[i]*m_s*m_s))

    fig2, (axup, axdown) = plt.subplots(2,1,sharex=True,
                                        gridspec_kw={'height_ratios': [2.1, 1]},
                                        layout='constrained')

    axdown.plot(beta_arr, C_arr[0])

    beta_arr, h_arr = np.meshgrid(beta_arr, h_arr)
    level = MaxNLocator(nbins=50).tick_values(C_arr.min(), C_arr.max())
    cf = axup.contourf(beta_arr, h_arr, C_arr, cmap="GnBu", levels=level)
    cbar = fig2.colorbar(cf)

    fig2.suptitle('Equilibrium CW Heat Capacity')
    axup.set_title(r'function of $\beta$ and $h$', fontsize=10.5)
    axdown.set_title('$h=0$', fontsize=10.5)

    cbar.set_label(r"$C /k_B$")

    axup.set_xlabel(r'$J\beta$')
    axdown.set_xlabel(r"$J\beta$")
    axdown.set_ylabel(r"$C/k_B$")

    axup.set_ylabel(r"$h$")

    plt.show()


def main():
    C_infty()


if __name__ == '__main__':
    main()
