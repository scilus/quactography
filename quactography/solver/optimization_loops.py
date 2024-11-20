from scipy.optimize import minimize
import numpy as np


# Minimization cost function
def cost_func(params, estimator, ansatz, hamiltonian):
    cost = (
        estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    )
    return cost


from scipy.optimize import minimize
import numpy as np


# Optimizer function
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
        # print(
        #     "Loop:",
        #     loop_count,
        #     "Iterations:",
        #     res.nit,
        #     "Cost:",
        #     new_cost,
        #     "Params found:",
        #     res.x,
        # )

        # Save same data to a text file:
        with open("params_iterations.txt", "a") as f:
            f.write(
                f"Loop: {loop_count}, Iterations: {res.nit}, Cost: {new_cost}, Params found: {res.x}\n"
            )

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

    # Add list of non-valid parameters to the text files
    with open("params_found_while_opt.txt", "a") as f:
        f.write(f"List of non-valid parameters: {no_valid_params}\n")

    return res, new_cost, previous_cost, x_0, loop_count, cost_history


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
    print("Starting refinement optimization...")

    def is_close_to_any(x_0, no_valid_params, tol=1e-2):
        return any(np.allclose(x_0, param, atol=tol) for param in no_valid_params)

    second_last_cost = last_cost

    while last_cost > 0 and num_refinement_loops > 0:
        # Generate new random parameters
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

        # If cost is increasing, stop the refinement loop and use the second-to-last result
        if potential_last_cost > last_cost:
            break

        # Update results if an improvement is found
        elif potential_last_cost < last_cost:
            second_last_cost = last_cost  # Update second-to-last cost
            last_cost = potential_last_cost
            res = potential_res  # Store the result
            previous_cost = potential_previous_cost
            x_0 = potential_x_0
            cost_history = potential_cost_history

        num_refinement_loops -= 1

    # Return the best result found before the last iteration
    return res, second_last_cost, previous_cost, x_0, loop_count, cost_history
