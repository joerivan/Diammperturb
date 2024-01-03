import numpy as np
import matplotlib.pyplot as plt


def main():

    # system
    a = 1
    V0 = 1

    npts = 501
    x = np.linspace(0, a, npts)

    # selecting ground/excited state
    # ground state, n = 1
    n = 1

    # transcendental solution
    I = 100  # number of iterations of fixed point equation
    psi, E = transcendental(x, a, V0, I, n)

    # plotting
    plt.figure(1, dpi=200)
    plt.plot(x, psi, label="trascendental", color="#785EF0")
    plt.xlabel('$x/a$ [-]')
    plt.ylabel('$\psi$ [-]')
    plt.grid(linestyle='--')

    # printing energies
    print("Energy (Ht): \n", E)


def transcendental(x, a=1, V0=1, I=10, n=1):
    """
    Calculate the wavefunction for transcendental technique up to Nth
    iterations of phi
    """
    # find the wave number
    if n == 0:
        raise Exception("Invalid state selected, in the ground state n=1")
    elif n % 2 == 1:
        # when wavefunction is effected by potential
        # iteratively find phase shift phi
        phi = [0]
        for i in range(I):
            phi.append(np.arctan(V0/(np.pi+2*phi[i]))+(n-1)/2*np.pi)
        phi = np.array(phi)

        k = np.pi + 2*phi[-1]
    elif n % 2 == 0:
        # when wavefunction is not effected by potential
        k = n*np.pi/a

    # find wavefunction values and normalise
    f = [transcendental_wf(xx, k, a) for xx in x]
    f = np.array(f)
    psi = abs(normalise(f, x))

    # determine energy
    E = k**2/2

    return psi, E


def transcendental_wf(x, k, a=1):
    """
    Calculate the wavefunction at point x for transcendental technique
    """

    if x < a/2:
        res = np.sin(k*x)
    elif x > a/2:
        res = np.sin(k*(a-x))
    elif x == a/2:
        res = np.sin(k*x)/2 + np.sin(k*(a-x))/2

    return res


def normalise(f, x):
    """
    Normalise function values f on domain x
    """

    y = f * np.sqrt(1/np.trapz(np.conjugate(f)*f, x))

    if np.mean(y) < 0:
        y = -y

    return y


if __name__ == '__main__':
    main()
