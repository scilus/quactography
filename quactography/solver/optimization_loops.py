from scipy.optimize import minimize
import numpy as np


# Minimization cost function
def cost_func(params, estimator, ansatz, hamiltonian):
    """
    Cost function to minimize for the optimization of the quantum circuit.

    Parameters
    ----------
    params : list
        List of parameters for the quantum circuit.
        (gamma, beta) angles depending on the number of layers.
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


def POWELL_loop_optimizer(
    loop_count,
    max_loops,
    previous_cost,
    epsilon,
    x_0,
    cost_history,
    estimator,
    ansatz,
    h,
):
    """
    Main optimization loop for the Powell optimizer.

    Parameters
    ----------
    loop_count : int
        Current loop count.
    max_loops : int
        Maximum number of loops.
    previous_cost : float
        Previous cost value.
    epsilon : float
        Tolerance for the convergence criterion.
    x_0 : list
        Initial parameters for the quantum circuit.
    cost_history : list
        List of cost values.
    estimator : Estimator from qiskit
        Estimator used to evaluate the cost function.
    ansatz : QuantumCircuit object from qiskit
        Quantum circuit used to generate the ansatz.
    h : Hamiltonian object from quactography library, Hamiltonian_qubit_edge
        Object containing total Hamiltonian function to minimize.
        Cost function in quantum Formalism.
    Returns
    -------
    res.x : list of floats
        Final optimized parameters.
    new_cost : float
        New cost value.
    previous_cost : float
        Previous cost value.
    x_0 : list of floats
        Initial parameters for the quantum circuit for the next iteration.
    loop_count : int
        Current loop count.
    cost_history : list of floats
        List of cost values updated.
    """
    # # Initialize list of invalid parameters
    no_valid_params = []

    while loop_count < max_loops:
        print("Loop:", loop_count)

        # Minimization with COBYLA or other method
        res = minimize(
            cost_func,
            x_0,
            args=(estimator, ansatz, h.total_hamiltonian),
            method="Powell",
            options={"maxiter": 10, "disp": False},
            tol=1e-5,
        )

        # Optimized cost
        new_cost = cost_func(res.x, estimator, ansatz, h.total_hamiltonian)
        print(
            "Loop:",
            loop_count,
            "Iterations:",
            res.nit,
            "Cost:",
            new_cost,
            "Params found:",
            res.x,
        )

        # # Save same data to a text file:
        # with open("params_iterations.txt", "a") as f:
        #     f.write(
        #         f"Loop: {loop_count}, Iterations: {res.nit},
        # Cost: {new_cost}, Params found: {res.x}\n"
        #     )

        # Check for convergence
        if abs(previous_cost - new_cost) < epsilon:
            print("Cost when convergence reached:", new_cost)

        # Update parameters if improvement found
        if new_cost < previous_cost:
            no_valid_params.append(res.x)
            x_0 = res.x + 0.4  # Adjust for next iteration
            previous_cost = new_cost
            cost_history.append(new_cost)
        else:
            no_valid_params.append(res.x)
            cost_history.append(new_cost)
            break

        loop_count += 1

    # print("List of non-valid parameters:", no_valid_params)

    # # Add list of non-valid parameters to the text files
    # with open("params_found_while_opt.txt", "a") as f:
    #     f.write(f"List of non-valid parameters: {no_valid_params}\n")
    # np.savez("params_found_while_opt.npz", no_valid_params=no_valid_params)

    return res.x, new_cost, previous_cost, x_0, loop_count, cost_history


def POWELL_refinement_optimization(
    loop_count,
    max_loops,
    estimator,
    ansatz,
    h,
    no_valid_params,
    epsilon,
    x_0,
    previous_cost,
    last_cost,
    cost_history,
    num_refinement_loops,
):
    """
    Refinement optimization loop for the Powell optimizer.
    Unused for now, issues to be fixed.

    Parameters
    ----------
    loop_count : int
        Current loop count.
    max_loops : int
        Maximum number of loops.
    estimator : Estimator from qiskit
        Estimator used to evaluate the cost function.
    ansatz : QuantumCircuit object from qiskit
        Quantum circuit used to generate the ansatz.
    h : Hamiltonian object from quactography library, Hamiltonian_qubit_edge
        Object containing total Hamiltonian function to minimize.
        Cost function in quantum Formalism.
    no_valid_params : list
        List of non-valid parameters.
    epsilon : float
        Tolerance for the convergence criterion.
    x_0 : list
        Initial parameters for the quantum circuit.
    previous_cost : float
        Previous cost value.
    last_cost : float
        Last cost value.
    cost_history : list
        List of cost values.
    num_refinement_loops : int
        Number of refinement loops to perform.
    Returns
    -------
    res_x : list
        Final optimized parameters.
    second_last_cost : float
        Second last cost value.
    previous_cost : float
        Previous cost value.
    x_0 : list
        Initial parameters for the quantum circuit for the next iteration.
    loop_count : int
        Current loop count.
    cost_history : list
        List of cost values updated.
    """
    print("Starting refinement optimization...")

    def is_close_to_any(x_0, no_valid_params, tol=1e-2):
        return any(np.allclose(x_0, param, atol=tol) for param in no_valid_params)

    second_last_cost = last_cost
    res_x = None

    while last_cost > 0 and num_refinement_loops > 0:
        # Generate new random parameters:
        while True:
            x_0 = np.random.uniform(0, np.pi, ansatz.num_parameters)
            if not is_close_to_any(x_0, no_valid_params):
                break

        # Run the optimizer:
        (
            potential_resx,
            potential_last_cost,
            potential_previous_cost,
            potential_x_0,
            loop_count,
            potential_cost_history,
        ) = POWELL_loop_optimizer(
            loop_count,
            max_loops,
            previous_cost,
            epsilon,
            x_0,
            cost_history,
            estimator,
            ansatz,
            h,
        )

        # Update results if an improvement is found:
        if potential_last_cost < last_cost:
            second_last_cost = last_cost
            last_cost = potential_last_cost
            res_x = potential_resx
            previous_cost = potential_previous_cost
            x_0 = potential_x_0
            cost_history = potential_cost_history
        else:
            break

        num_refinement_loops -= 1

    # If no valid result was found, use the last known good parameters:
    if res_x is None:
        res_x = x_0

    return res_x, second_last_cost, previous_cost, x_0, loop_count, cost_history
