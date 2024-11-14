from scipy.optimize import minimize
import numpy as np


# Minimization cost function
def cost_func(params, estimator, ansatz, hamiltonian):
    cost = (
        estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    )
    return cost


# Optimizer function
def COBYLA_loop_optimizer(
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
    # # Initialize list of invalid parameters
    no_valid_params = []

    # # Initialize parameters for the optimizer
    # x_0 = np.zeros(ansatz.num_parameters)
    # epsilon = epsilon
    # previous_cost = np.inf
    # cost_history = []
    # loop_count = 0
    # max_loops = 30
    # no_valid_params = []
    while loop_count < max_loops:
        print("Loop:", loop_count)

        # Minimization with COBYLA or other method
        res = minimize(
            cost_func,
            x_0,
            args=(estimator, ansatz, h.total_hamiltonian),
            method="COBYLA",
            options={"maxiter": 500, "disp": False},
            tol=1e-5,
        )

        # Optimized cost
        new_cost = cost_func(res.x, estimator, ansatz, h.total_hamiltonian)
        print("Loop:", loop_count, "Iterations:", res.nfev, "Cost:", new_cost)

        # Check for convergence
        if abs(previous_cost - new_cost) < epsilon:
            print("Cost when convergence reached:", new_cost)

        # Update starting point and previous cost if improvement found
        if new_cost < previous_cost:
            no_valid_params.append(res.x)
            x_0 = res.x + 0.4
            previous_cost = new_cost
            cost_history.append(new_cost)
        else:
            no_valid_params.append(res.x)
            cost_history.append(new_cost)
            break

        if loop_count == max_loops - 1:
            print("Max loops reached")
            print("Cost when max loops reached:", new_cost)
            break

        loop_count += 1

    print("List of non-valid parameters:", no_valid_params)
    return res, new_cost, previous_cost, x_0, loop_count, cost_history


def COBYLA_refinement_optimization(
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
    # Helper function to check if a parameter is close to any of the non valid parameters
    def is_close_to_any(x_0, no_valid_params, tol=1e-2):
        return any(np.allclose(x_0, param, atol=tol) for param in no_valid_params)

    # Main refinement loop:
    while last_cost > h.graph.min_weight and num_refinement_loops > 0:
        # Set x_0 to random parameters not in the list of non valid parameters
        while True:
            x_0 = np.random.uniform(0, np.pi, ansatz.num_parameters)
            if not is_close_to_any(x_0, no_valid_params):
                break

        # Run the optimizer
        (
            potential_res,
            potential_last_cost,
            potential_previous_cost,
            potential_x_0,
            loop_count,
            potential_cost_history,
        ) = COBYLA_loop_optimizer(
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
        # If cost is growing, cancel the refinement loop, return the last result
        if potential_last_cost > last_cost:
            break
        if potential_last_cost < last_cost:
            last_cost = potential_last_cost
            res = potential_res
            previous_cost = potential_previous_cost
            x_0 = potential_x_0
            cost_history = potential_cost_history

        num_refinement_loops -= 1
