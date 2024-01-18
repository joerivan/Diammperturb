import numpy as np
import matplotlib.pyplot as plt
import os


def main():

    # settings folder here the file runs
    folder = os.path.dirname(__file__)

    # system
    a = 1

    npts = 501
    x = np.linspace(0, a, npts)

    # selecting ground/excited state
    # ground state, n = 1
    n = 1

    # exact solution
    psi_exact, E_exact = exact(x, a, n)

    # linear variational method
    E_lvm = []
    N_list = grid(20, 600)
    for N in N_list:
        psi_lvm, e_lvm = lvm(x, a, n, N)
        E_lvm.append(e_lvm)

    # pertrubation theory
    E_pt = []
    O_list = np.arange(1, 7)
    for order in O_list:
        e_pt = perturb(order)
        E_pt.append(e_pt)

    # printing energies
    print(f"Exact energy:\n{E_exact} Ht\n")
    print("Linear variational method energies:")
    for i, N in enumerate(N_list):
        print(f"{E_lvm[i]} Ht   N={N}")
    print("\nPerturbation theory:")
    for i, O in enumerate(O_list):
        print(f"{E_pt[i]} Ht   order={O}")

    # plotting
    f, (ax1, ax2) = plt.subplots(1, 2, dpi=400, sharey=True, figsize=(10, 4.8))
    ax1.plot(N_list, E_lvm, '-o', color='#DC267F')
    ax1.set_ylabel('E [Ht]')
    ax1.set_xlabel('N [-]')
    ax1.set_xlim([N_list[0], N_list[-1]])
    ax1.grid(linestyle='--')
    ax1.set_title('Linear variational method')
    ax1.axhline(E_exact, color='#000000', linewidth=1)

    ax2.plot(O_list, E_pt, '-o', color='#648FFF')
    ax2.set_ylabel('E [Ht]')
    ax2.set_xlabel('correction order')
    ax2.set_xlim([O_list[0], O_list[-1]])
    ax2.grid(linestyle='--')
    ax2.set_title('Perturbation theory')
    ax2.axhline(E_exact, color='#000000', linewidth=1)
    ax2.yaxis.set_tick_params(labelleft=True)

    plt.tight_layout()

    # saving plot
    plt.savefig(os.path.join(folder, 'comparison_lvm_pt.svg'))


def perturb(order):

    E = [4.934802201,
         6.934802201,
         6.732159833,
         6.746740236,
         6.746197706,
         6.746173907,
         6.746178792,
         6.746178595,
         6.746178569,
         6.746178573,
         6.746178573]

    return E[order]


def lvm(x, a, n, N):
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
            V[i, j] = 2. / a * np.sin((i+1) * np.pi / 2.) * \
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


def exact(x, a=1, n=1, tol=1e-20):
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
            phi.append(np.arctan(1/(np.pi+2*phi[i]))+(n-1)/2*np.pi)
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


def grid(N, max_x):
    """
    logspace integer grid for linear variational methodwith roughly 
    N points and max integer of max_x
    """
    # logspace
    z = np.geomspace(1, max_x, N)

    # making integers
    r = [int(zz) for zz in z]

    # sifting out duplicates and sorting
    r = list(set(r))
    r.sort()

    return r


if __name__ == '__main__':
    main()
