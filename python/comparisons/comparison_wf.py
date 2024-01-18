import numpy as np
import matplotlib.pyplot as plt
import os


def main():

    # settings folder hwere the file runs
    folder = os.path.dirname(__file__)

    # system
    a = 1
    V0 = 1

    npts = 10001
    x = np.linspace(0, a, npts)

    # selecting ground/excited state
    # ground state, n = 1
    n = 1

    # number of basis functions
    N = 10000

    # exact solution
    psi_exact, E_exact = exact(x, a, V0, n)

    # linear variational principle
    psi_lvm, E_lvm = lvm(x, a, V0, n, N)

    # plotting
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, dpi=400, figsize=(10, 4.8))
    ax1.plot(x, psi_exact, color="#785EF0")
    ax1.set_xlabel('$x/a$ [-]')
    ax1.set_ylabel(r'$\psi$ [-]')
    ax1.grid(linestyle='--')

    ax2.plot(x, psi_exact-psi_lvm, color="#FE6100")
    ax2.set_xlabel('$x/a$ [-]')
    ax2.set_ylabel('Residuals [-]')
    ax2.grid(linestyle='--')

    plt.tight_layout()

    # printing energies
    print(f"Exact energy:\n{E_exact} Ht\n")
    print(f"Linear variational method (N={N}) energy:\n{E_lvm} Ht\n")
    print(f"Energy difference:\n{abs(E_exact-E_lvm)} Ht")

    # saving plot
    plt.savefig(os.path.join(folder, 'comparison_exact_lvm.svg'))


def lvm(x, a, V0, n, N):
    """
    Find the wavefunction and energies using linear variational method with
    N basis fucntions
    """

    # build kinetic matrix
    T = np.diag([(nn+1)**2 * np.pi**2 / 2. / a**2 for nn in range(0, N)])

    # build potential matrix
    V = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            # potential matrix element due to Dirac potential
            V[i, j] = V0 * 2. / a * np.sin((i+1) * np.pi / 2.) * \
                np.sin((j+1) * np.pi / 2.)

    # add matrices together and perform eigenvalue decomposition
    H = T + V
    e, v = np.linalg.eigh(H)

    # print resultant function
    ywf = [v[nn, n-1] * np.sqrt(2./a) * np.sin((nn+1) * np.pi / a * x)
           for nn in range(0, N)]
    psi = normalise(np.einsum('ij->j', ywf), x)
    E = e[n-1]

    return psi, E


def exact(x, a=1, V0=1, n=1, tol=1e-20):
    """
    Calculate the exact wavefunction
    """
    # find the wave number
    if n == 0:
        raise Exception("Invalid state selected, in the ground state n=1")
    elif n % 2 == 1:
        # when wavefunction is effected by potential
        # find phase shift phi

        i = 0
        diff = 1
        phi = [0]

        while diff > tol:
            phi.append(np.arctan(V0/(np.pi+2*phi[i]))+(n-1)/2*np.pi)
            i += 1
            diff = abs(phi[i-1] - phi[i])
        phi = np.array(phi)

        k = np.pi + 2*phi[-1]

    elif n % 2 == 0:
        # when wavefunction is not effected by potential
        k = n*np.pi/a

    # find wavefunction values and normalise
    f = [exact_wf(xx, k, a) for xx in x]
    f = np.array(f)
    psi = normalise(f, x)

    # determine energy
    E = k**2/2

    return psi, E


def exact_wf(x, k, a=1):
    """
    Calculate the exact wavefunction at point x
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
