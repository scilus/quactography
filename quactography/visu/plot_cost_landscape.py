import matplotlib.pyplot as plt
import numpy as np
from functools import partial


# Minimization cost function
def cost_func(params, estimator, ansatz, hamiltonian):
    """
    Cost function to minimize for the optimization of the quantum circuit.

    Parameters
    ----------
    params : list
        List of parameters for the quantum circuit. (gamma, beta) angles depending on the number of layers.
    estimator : Estimator from qiskit
        Estimator used to evaluate the cost function.
    ansatz : QuantumCircuit object from qiskit
        Quantum circuit used to generate the ansatz.
    hamiltonian : PauliSum object from qiskit
        Hamiltonian to minimize. Cost function in quantum Formalism.
    Returns
    -------
    cost : float
        Value of the cost function for the given parameters.
    """
    cost = (
        estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    )
    return cost


def plt_cost_func(estimator, ansatz, h):
    """
    Plot the cost function landscape for the given Hamiltonian. 
    The cost function is evaluated for all possible values of gamma and beta for only one layer.(reps=1)

    Parameters
    ----------
    estimator : Estimator from qiskit
        Estimator used to evaluate the cost function.
    ansatz : QuantumCircuit object from qiskit
        Quantum circuit used to generate the ansatz.
    h : Hamiltonian object from quactography library, Hamiltonian_qubit_edge 
        Object containing total Hamiltonian function to minimize. Cost function in quantum Formalism.

    Returns
    -------
    fig : image
        Figure object containing the plots.
    ax1 : image
        3D plot of the cost function landscape.
    ax2 : image
        2D contour plot of the cost function landscape.
    """
    cost_func_with_args = partial(
        cost_func,
        estimator=estimator,
        ansatz=ansatz,
        hamiltonian=h.total_hamiltonian,
    )
    gamma_range = np.arange(0, 2 * np.pi, 0.2)
    beta_range = np.arange(0, np.pi, 0.2)

    # meshgrid:
    gamma, beta = np.meshgrid(gamma_range, beta_range)
    # Compute z by evaluating the cost function for each pair (gamma, beta)
    z = np.zeros_like(gamma)
    for i in range(gamma.shape[0]):
        for j in range(gamma.shape[1]):
            params = [gamma[i, j], beta[i, j]]
            z[i, j] = cost_func_with_args(params)

    # 3D Surface Plot
    fig = plt.figure(figsize=(12, 6))

    # 3D plot
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(  # type: ignore
        gamma, beta, z, cmap="jet", linewidth=0, antialiased=False
    )
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_xlabel("Gamma")
    ax1.set_ylabel("Beta")
     # ax1.set_zlabel("Cost Function Value")  # type: ignore
    ax1.set_title("3D Cost Function Surface")

    # 2D Contour Plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(gamma, beta, z, levels=50, cmap="jet")
    fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)
    ax2.set_xlabel("Gamma")
    ax2.set_ylabel("Beta")
    ax2.set_title("2D Cost Function Contour")

    # Save and show the plots
    plt.tight_layout()
    # plt.savefig("cost_function_landscape.png")
    # plt.show()
    return fig, ax1, ax2
